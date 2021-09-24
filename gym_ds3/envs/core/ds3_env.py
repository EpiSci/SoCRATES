import os
from os.path import dirname, abspath

import gym
import matplotlib.pyplot as plt
import numpy as np
import simpy

from gym_ds3.envs.core import processing_element, job_generator, DASH_Sim_core
from gym_ds3.envs.core.env_storage import EnvStorage
from gym_ds3.envs.misc import DASH_SoC_parser, job_parser
from gym_ds3.envs.power import DTPM, DTPM_policies
from gym_ds3.envs.misc.action_map import *
from gym_ds3.envs.utils.helper_envs import str_to_list


class DS3GymEnv(gym.Env):
    def __init__(self):
        self.updated = False
        self.pe_return_list = []

        self.curr_comp_jobs = 0
        
    def reset(self, args):
        self.args = args
        
        self.env = simpy.Environment()
        self.jobs, self.resource_matrix, self.env_storage = self._create_env_storage()
        self.comm_band = self.resource_matrix.comm_band
        self.resource_matrix_list = self.resource_matrix.list
        self.num_pes = len(self.resource_matrix_list) - 1
        
        self.run_mode = args.run_mode
        self.scheduler_name = args.scheduler_name
        
        if self.run_mode == 'step':
            self.rts = {}

        self.job_dags = [jd for jd in self.env_storage.job_dags if not jd.is_completed]

        self.gym_step = 0

        self.prev_time = 0

        self.curr_comp_jobs = 0

    def _reset_core(self, scale, scheduler):
        # Construct the processing elements in the target DSSoC
        self.pes = [
            processing_element.PE(
                self.env, self,
                self.resource_matrix_list[i].type, self.resource_matrix_list[i].name,
                self.resource_matrix_list[i].ID, self.resource_matrix_list[i].capacity,
                self.args.resource_std
            )
            for i in range(len(self.resource_matrix.list))
        ]
        
        self.job_gen = job_generator.JobGenerator(
            self.args, scale, self, self.env, self.pes,
            (self.jobs, self.resource_matrix),
            self.env_storage)
        self.env.process(self.job_gen.run())
        
        self.simulator = DASH_Sim_core.SimulationManager(
            self, self.env, (self.jobs, self.resource_matrix), self.pes, scheduler)

        if self.run_mode == 'run':
            self.env.process(self.simulator.run(self.env_storage))
            
        elif self.run_mode == 'step':
            self.event = self.env.event()
        
            while not self.env_storage.TaskQueues.ready.list:
                self.env.step()

            for task in self.env_storage.TaskQueues.ready.list:
                self.rts[task.ID] = (task.base_ID, task.PE_ID)
                
                self.DTPM_module = DTPM.DTPMmodule(self.args, self.env, self.env_storage, self.resource_matrix, self.pes)

                for resource, current_PE in zip(self.resource_matrix.list, self.pes):
                    DTPM_policies.initialize_frequency(resource, current_PE)
                
                if self.env.now % self.args.sampling_rate == 0:
                    # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                    self.DTPM_module.evaluate_idle_PEs()

        if self.run_mode == 'step':
            self.env.process(self._timeout(1))

    def update_reward(self, now, reward):
        self.env_storage.buf.exp['reward'][-1] += reward
        self.env_storage.buf.exp['wall_time'][-1] += now

    def _timeout(self, num):
        while True:
            yield self.env.timeout(num) # map(lambda x, y: x | y, self.pe_return_list)

    def _event(self):
        self.event.succeed()
        self.event = self.env.event()

    def run(self, num_gradient_steps, scale, scheduler):
        self.episode = num_gradient_steps
        self._reset_core(scale, scheduler)
        
        if self.args.run_mode == 'run':
            self.env.run(until=self.args.simulation_length)
            inj_jobs, cum_exec_time, comp_jobs, total_energy_consumption, edp, latency = self.compute_latency()
            print('[TRAIN] ep: {} | {}/{} comp/inj jobs | {} latency'.format(
                int(num_gradient_steps + 1), comp_jobs, inj_jobs, latency))
        elif self.args.run_mode == 'step':
            done = False
            
            while not done:
                action = scheduler.schedule(self)
                _, _, done, _ = self.step(action)
            
            inj_jobs, cum_exec_time, comp_jobs, total_energy_consumption, edp, latency = self.compute_latency()
            print('[TRAIN] ep: {} | {}/{} comp/inj jobs | {} latency'.format(
                int(num_gradient_steps + 1), comp_jobs, inj_jobs, latency))
    
        return inj_jobs, comp_jobs, cum_exec_time, total_energy_consumption, edp, latency

    def select_action(self, mapped_tasks):
        for task in self.env_storage.TaskQueues.ready.list:
            if task.ID in list(mapped_tasks.keys()):
                task.PE_ID = mapped_tasks[task.ID][0]
                task.dynamic_dependencies = mapped_tasks[task.ID][2]

    def step(self, action):

        if self.rts and action is not None:
            self.select_action(action)

            self.rts = {}

            obs = (self.env_storage, self.pes, self.env.now)
            done = self._get_done()
            reward = self._get_reward(self.env_storage)

            return obs, reward, done, {}

        if (not self.updated) and (not self.rts):
            # Update the execution queue based on task's info
            self.simulator.update_execution_queue(self.env_storage, self.env_storage.TaskQueues.ready.list)
            self.updated = True

        # Initialize $remove_from_executable which will populate tasks to be removed from the executable queue
        remove_from_executable = []

        # Go over each task in the executable queue
        if len(self.env_storage.TaskQueues.executable.list) != 0:
            for executable_task in self.env_storage.TaskQueues.executable.list:
                is_time_to_execute = (executable_task.time_stamp <= self.env.now)
                PE_has_capacity = (len(self.pes[executable_task.PE_ID].queue) < self.pes[executable_task.PE_ID].capacity)
                task_has_assignment = (executable_task.PE_ID != -1)

                dynamic_dependencies_met = True

                dependencies_completed = []
                for dynamic_dependency in executable_task.dynamic_dependencies:
                    dependencies_completed = dependencies_completed + list(
                        filter(lambda completed_task: completed_task.ID == dynamic_dependency,
                               self.env_storage.TaskQueues.completed.list))
                if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                    dynamic_dependencies_met = False

                if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met and task_has_assignment:
                    self.pes[executable_task.PE_ID].queue.append(executable_task)

                    current_resource = self.resource_matrix.list[executable_task.PE_ID]
                    executable_task.wait_time = max(0, self.env.now - executable_task.ready_time)

                    self.env.process(self.pes[executable_task.PE_ID].execute(
                        self.simulator, executable_task, current_resource, self.env_storage, self.DTPM_module))

                    remove_from_executable.append(executable_task)

        # Remove the tasks from executable queue that have been executed by a resource
        for task in remove_from_executable:
            self.env_storage.TaskQueues.executable.list.remove(task)

        for pe in self.pe_return_list:
            if type(pe._value) is bool and pe._value:
                self.pe_return_list.remove(pe)

        self.env.step()
        for pe in self.pes:
            if pe.task:
                if pe.task_expected_total_time + pe.task.start_time == self.env.now:
                    self.env.step()

        # Update rts
        if self.env_storage.TaskQueues.ready.list:
            for task in self.env_storage.TaskQueues.ready.list:
                self.rts[task.ID] = (task.base_ID, task.PE_ID)

        obs = (self.env_storage, self.pes, self.env.now)
        done = self._get_done()
        reward = self._get_reward(self.env_storage)
        return obs, reward, done, {}

    def _get_reward(self, env_storage):
        
        curr_time = self.now()
        reward = 0
        
        if self.args.learn_obj == 'duration':
            n_comp_jobs = len(self.env_storage.completed_job_dags)
            if n_comp_jobs == 0:
                comp_factor = 1
            else:
                comp_factor = 1 / n_comp_jobs

            for job_dag in list(self.job_dags):
                reward -= (min(job_dag.completion_time, curr_time) - \
                            min(job_dag.start_inject_time, self.prev_time)) * \
                        comp_factor
            
            reward = reward / 10000.
        
        elif self.args.learn_obj == 'makespan':
            reward = -1 * (curr_time - self.prev_time)

        elif self.args.learn_obj == 'latency':
            reward -= (1 - (self.env_storage.results.cumulative_exe_time / (self.env_storage.results.completed_jobs + 1e-7)))
        
        elif self.args.learn_obj == 'compbonus':
            bonus = 0
            if len(env_storage.completed_job_dags) != self.curr_comp_jobs:
                curr_jobs = len(env_storage.completed_job_dags) - self.curr_comp_jobs
                self.curr_comp_jobs = len(env_storage.completed_job_dags)
                bonus += (50 * curr_jobs)
            
            reward = -0.5 + bonus
        
        self.prev_time = curr_time
        
        return reward

    def _get_done(self):
        self.done = self.env.now > (self.args.simulation_length - 2)
        return self.done

    def now(self):
        return self.env.now

    def render(self, outs, labels, data_path=None):
        fig = plt.figure(figsize=(24, 12), dpi=200)
        data = []
        for i in range(len(outs)):
            data.append(plt.subplot2grid((2, 4), (0, i), colspan=1, rowspan=1))

        for idx in range(len(data)):
            out = data[idx].matshow([outs[idx][:, i] for i in range(len(outs[idx][0]))],
                                    aspect='auto', cmap='jet', vmin=-1, vmax=1)
            for edge, spine in data[idx].spines.items():
                spine.set_visible(False)
            data[idx].set_xticks(np.arange(len(outs[idx]), step=10))
            data[idx].set_yticks(np.arange(len(outs[idx][0]), step=2))
            data[idx].set_xticklabels(np.arange(len(outs[idx])), fontsize=5)
            data[idx].set_yticklabels(np.arange(len(outs[idx])), fontsize=3)
            data[idx].tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

            cb = fig.colorbar(out, ax=data[idx])
            cb.ax.tick_params(labelsize=5)

            data[idx].set_xticks(np.arange(len(outs[idx])) + 0.5, minor=True)
            data[idx].grid(which='minor', color='w', linewidth=0.5)
            data[idx].tick_params(which="both", bottom=True, left=True, top=False, labeltop=False)
            data[idx].tick_params(which="minor", bottom=False, left=False, top=False, labeltop=False, labelbottom=False)

            data[idx].set_xlabel('Number of {}'.format(labels[idx]), fontsize=6)
            data[idx].set_ylabel('Number of {} feature'.format(labels[idx]), fontsize=6)
            data[idx].set_title('{} state'.format(labels[idx]), fontsize=7)

        if data_path is None:
            plt.show()
        else:
            plt.savefig(data_path + '/step_{}_{}.png'.format(self.episode, self.now()), dpi=300)

        plt.close('all')

    def compute_latency(self):
        total_inj_jobs = self.env_storage.results.injected_jobs
        total_cum_exec_time = self.env_storage.results.cumulative_exe_time
        total_comp_jobs = self.env_storage.results.completed_jobs
        total_energy_consumption = self.env_storage.results.cumulative_energy_consumption
        
        edp = self.env_storage.results.cumulative_exe_time * self.env_storage.results.cumulative_energy_consumption
        avg_latency = total_cum_exec_time / (total_comp_jobs + 1e-7)

        return total_inj_jobs, total_cum_exec_time, total_comp_jobs, total_energy_consumption, edp, avg_latency

    def _create_env_storage(self):
        env_storage = EnvStorage()

        # Instantiate the ResourceManager object that contains all the resources in the target DSSoC
        resource_matrix = env_storage.ResourceManager()  # This line generates an empty resource matrix
        resource_file_list = str_to_list(
            os.path.join(dirname(dirname(dirname(abspath(__file__)))) + '/' + 'data' + '/' + self.args.resource_profile))
        for resource_file in resource_file_list:
            DASH_SoC_parser.resource_parse(self.args, resource_matrix, resource_file)

        # Instantiate the ApplicationManager object that contains all the jobs in the target DSSoC
        jobs = env_storage.ApplicationManager()  # This line generates an empty list for all jobs
        job_files_list = [os.path.join(dirname(dirname(dirname(abspath(__file__))))) + '/' + 'data' + '/' + f
                          for f in str_to_list(self.args.job_profile)]
        for job_file in job_files_list:
            job_parser.job_parse(jobs, job_file)  # Parse the input job file to populate the job list

        # Initially none of the tasks are outstanding
        env_storage.TaskQueues.outstanding \
            = env_storage.TaskManager()  # List of *all* tasks waiting to be processed
        # Initially none of the tasks are completed
        env_storage.TaskQueues.completed \
            = env_storage.TaskManager()  # List of completed tasks
        # Initially none of the tasks are running on the pes
        env_storage.TaskQueues.running \
            = env_storage.TaskManager()  # List of currently running tasks
        # Initially none of the tasks are completed
        env_storage.TaskQueues.ready \
            = env_storage.TaskManager()  # List of tasks that are ready for processing
        # Initially none of the tasks are in wait ready queue
        env_storage.TaskQueues.wait_ready \
            = env_storage.TaskManager()  # List of tasks that are waiting for being ready for processing
        # Initially none of the tasks are executable
        env_storage.TaskQueues.executable \
            = env_storage.TaskManager()  # List of tasks that are ready for execution

        env_storage.results = env_storage.PerfStatics()

        return jobs, resource_matrix, env_storage


def record_value(d, key, value):
    for k, v in d.items():
        if k == key:
            d[k] = value


def remove_value_from_dict(d, val):
    for elem in d.items():
        if elem[1] == val:
            k = elem[0]
    d.pop(k, 'None')


def decrease_value(d, minus):
    for key, value in d.items():
        d[key] = value - minus
        