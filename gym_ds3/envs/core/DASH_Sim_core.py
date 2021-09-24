'''
Description: This file contains the simulation core that handles the simulation events.
'''
from collections import namedtuple
TaskInformation = namedtuple(
    'TaskInformation', 
    ['state_at_scheduling', 'task_id', 'task_base_id', 
     'probs_idx', 'action_at_scheduling', 'timestep_of_scheduling']
)
import numpy as np

import torch

from gym_ds3.envs.power import DTPM as DTPM
from gym_ds3.envs.power import DTPM_policies as DTPM_policies


class SimulationManager:
    '''
    Define the SimulationManager class to handle the simulation events.
    '''
    def __init__(self, gym_env, env, input_data, PE_list, scheduler):
        '''
        env: Pointer to the current simulation environment
        scheduler: Pointer to the DASH_scheduler
        PE_list: The PEs available in the current SoC
        jobs: The list of all jobs given to DASH-Sim
        resource_matrix: The data structure that defines power/performance
            characteristics of the PEs for each supported task
        '''
        self.gym_env = gym_env
        self.env = env
        self.jobs = input_data[0]
        self.resource_matrix = input_data[1]
        self.PEs = PE_list
        self.scheduler = scheduler
        
        self.args = self.gym_env.args
        self.max_num_jobs = self.args.max_num_jobs
        self.sampling_rate = self.args.sampling_rate
        self.scheduler_name = self.args.scheduler_name

        # used for SoCRATES, SCARL schedulers
        self.task_completed = []
        self.rewards_by_flops = []
        self.infos = []
        
    # As the simulation proceeds, tasks are being processed.
    # We need to update the ready tasks queue after completion of each task
    def update_ready_queue(self, env_storage, completed_task):
        '''
        This function updates the env_storage.TaskQueues.ready after one task is completed.
        '''
        # completed_task is the task whose processing is just completed
        # Add completed task to the completed tasks queue
        env_storage.TaskQueues.completed.list.append(completed_task)

        # Remove the completed task from the queue of the PE
        for task in self.PEs[completed_task.PE_ID].queue:
            if task.ID == completed_task.ID:
                self.PEs[task.PE_ID].queue.remove(task)

        # Remove the completed task from the currently running queue
        # if completed_task.ID in env_storage.TaskQueues.running.list:
        env_storage.TaskQueues.running.list.remove(completed_task)

        # Initialize $remove_from_outstanding_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_outstanding_queue = []

        # Initialize $to_memory_comm_time which will be communication time to
        # memory for data from a predecessor task to a outstanding task
        to_memory_comm_time = -1

        jobs_list = self.jobs.list

        job_ID = -1
        for ind, job in enumerate(jobs_list):
            if job.name == completed_task.jobname:  # and (job.task_list[0].jobID == completed_task.jobID):
                job_ID = ind

        # Check if the dependency of any outstanding task is cleared
        # We need to move them to the ready queue
        for i, outstanding_task in enumerate(env_storage.TaskQueues.outstanding.list):  # Go over each outstanding task
            if (completed_task.ID in outstanding_task.predecessors):  # if the completed task is one of the predecessors
                outstanding_task.predecessors.remove(completed_task.ID)  # Clear this predecessor

                if self.args.shared_memory:
                    # Get the communication time to memory for data from a
                    # predecessor task to a outstanding task
                    comm_vol = jobs_list[job_ID].comm_vol[completed_task.base_ID, outstanding_task.base_ID]
                    comm_band = self.resource_matrix.comm_band[completed_task.PE_ID, self.resource_matrix.list[-1].ID]
                    to_memory_comm_time = int(comm_vol / comm_band)  # Communication time from a PE to memory

                    if self.args.verbose:
                        print('[D] Time %d: Data from task %d for task %d will be sent to memory in %d us'
                              % (self.env.now, completed_task.ID, outstanding_task.ID, to_memory_comm_time))

                    # Based on this communication time, this outstanding task
                    # will be added to the ready queue. That is why, keep track of
                    # all communication times required for a task in the list
                    # $ready_wait_times
                    outstanding_task.ready_wait_times.append(to_memory_comm_time + self.env.now)

            no_predecessors = (len(outstanding_task.predecessors) == 0)  # Check if this was the last dependency
            currently_running = (outstanding_task in  # if the task is in the running queue,
                                 env_storage.TaskQueues.running.list)  # We should not put it back to the ready queue
            not_in_ready_queue = not(outstanding_task in  # If this task is already in the ready queue,
                                  env_storage.TaskQueues.ready.list)  # We should not append another copy

            if (no_predecessors and not (currently_running) and not_in_ready_queue):
                if self.args.shared_memory:
                    # if shared memory is utilized for communication, then
                    # the outstanding task will wait for a certain amount time
                    # (till the $time_stamp)for being added into the ready queue
                    env_storage.TaskQueues.wait_ready.list.append(outstanding_task)
                    if self.args.verbose:
                        print('[I] Time %d: Task %d ready times due to memory communication of its predecessors are'
                              % (self.env.now, outstanding_task.ID))
                        print('%12s' % (''), outstanding_task.ready_wait_times)
                    env_storage.TaskQueues.wait_ready.list[-1].time_stamp = max(outstanding_task.ready_wait_times)

                else: # if PE to PE communication is utilized
                    next_task = env_storage.TaskQueues.outstanding.list[i]
                    next_task.ready_time = self.env.now
                    env_storage.TaskQueues.ready.list.append(next_task)  # Add the task to the ready queue immediately
                    
                remove_from_outstanding_queue.append(outstanding_task)

        # Remove the tasks from outstanding queue that have been moved to ready queue
        for task in remove_from_outstanding_queue:
            env_storage.TaskQueues.outstanding.list.remove(task)

        # At the end of this function:
        # Newly processed $completed_task is added to the completed tasks
        # outstanding tasks with no dependencies are added to the ready queue
        # based on the communication mode and then, they are removed from
        # the outstanding queue

    def update_execution_queue(self, env_storage, ready_list):
        '''
        This function updates the env_storage.TaskQueues.executable if one task is ready
        for execution but waiting for the communication time, either between
        memory and a PE, or between two PEs (based on the communication mode)
        '''
        # Initialize $remove_from_ready_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_ready_queue = []

        # Initialize $from_memory_comm_time which will be communication time
        # for data from memory to a PE
        from_memory_comm_time = -1

        # Initialize $PE_to_PE_comm_time which will be communication time
        # for data from a PE to another PE
        PE_to_PE_comm_time = -1

        job_ID = -1

        jobs_list = self.jobs.list
        for ready_task in ready_list:
            if ready_task.PE_ID == -1:
                continue

            for ind, job in enumerate(jobs_list):
                # if job.task_list[0].jobID == ready_task.jobID:
                if job.name == ready_task.jobname:
                    job_ID = ind

            job_task_list = jobs_list[job_ID].task_list
            for task in job_task_list:
                compare_ID = ready_task.base_ID

                if compare_ID == task.ID:
                    if ready_task.head == True:
                        # if a task is the leading task of a job
                        # then it can start immediately since it has no predecessor
                        ready_task.PE_to_PE_wait_time.append(self.env.now)
                        ready_task.execution_wait_times.append(self.env.now)

                    for predecessor in task.predecessors:
                        if task.ID == ready_task.ID:
                            ready_task.predecessors = task.predecessors

                        # data required from the predecessor for $ready_task0
                        comm_vol = jobs_list[job_ID].comm_vol[predecessor, ready_task.base_ID]

                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + ready_task.ID - ready_task.base_ID

                        # Initialize following two variables which will be used if PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1

                        if self.args.shared_memory:
                            # Compute the memory to PE communication time
                            comm_band = self.resource_matrix.comm_band[self.resource_matrix.list[-1].ID, ready_task.PE_ID]
                            from_memory_comm_time = int(comm_vol / comm_band)
                            if self.args.verbose:
                                print('[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                                    % (self.env.now, ready_task.ID, real_predecessor_ID, ready_task.PE_ID, from_memory_comm_time))
                            ready_task.execution_wait_times.append(from_memory_comm_time + self.env.now)

                        else:  # PE_to_PE mode
                            # Compute the PE to PE communication time
                            for completed in env_storage.TaskQueues.completed.list:  # Find the real predecessor id in the completed task
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    ready_task.predecessors_finish_times.append(predecessor_finish_time)
                                    ready_task.predecessor_PE_IDs.append(predecessor_PE_ID)
                            comm_band = self.resource_matrix.comm_band[predecessor_PE_ID, ready_task.PE_ID]

                            PE_to_PE_comm_time = int(comm_vol / comm_band)
                            ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)

                            if self.args.verbose:
                                print('[D] Time %d: Data transfer from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                    % (self.env.now, predecessor_PE_ID, ready_task.PE_ID,
                                       ready_task.ID, real_predecessor_ID, ready_task.PE_to_PE_wait_time[-1]))

                    if self.args.verbose:
                        if self.args.shared_memory:
                            print('[I] Time %d: Task %d execution ready time(s) due to communication between memory and PE-%s are'
                              % (self.env.now, ready_task.ID, ready_task.PE_ID))
                            print('%12s' % (''), ready_task.execution_wait_times)    
                        else:
                            print('[I] Time %d: Task %d execution ready times due to communication between PEs are'
                                % (self.env.now, ready_task.ID))
                            print('%12s' % (''), ready_task.PE_to_PE_wait_time)

                    # Populate all ready tasks in executable with a time stamp
                    # which will show when a task is ready for execution
                    env_storage.TaskQueues.executable.list.append(ready_task)
                    remove_from_ready_queue.append(ready_task)
                    if self.args.shared_memory:
                        env_storage.TaskQueues.executable.list[-1].time_stamp = max(ready_task.execution_wait_times)
                    else:
                        env_storage.TaskQueues.executable.list[-1].time_stamp = max(ready_task.PE_to_PE_wait_time)

        # Remove the tasks from ready queue that have been moved to executable queue
        for task in remove_from_ready_queue:
            env_storage.TaskQueues.ready.list.remove(task)

    def update_completed_queue(self, env_storage):
        '''
        This function updates the env_storage.TaskQueues.completed
        '''
        # Be careful about this function when there are diff jobs in the system
        # reorder tasks based on their job IDs
        env_storage.TaskQueues.completed.list.sort(key=lambda x: x.jobID, reverse=False)

        first_task_jobID = env_storage.TaskQueues.completed.list[0].jobID
        last_task_jobID = env_storage.TaskQueues.completed.list[-1].jobID

        if ((last_task_jobID - first_task_jobID) > 15):
            for i, task in enumerate(env_storage.TaskQueues.completed.list):
                if (task.jobID == first_task_jobID):
                    del env_storage.TaskQueues.completed.list[i]

    def select_action(self, env_storage, mapped_tasks):
        for task in env_storage.TaskQueues.ready.list:
            if task.ID in list(mapped_tasks.keys()):
                task.PE_ID = mapped_tasks[task.ID][0]
                task.dynamic_dependencies = mapped_tasks[task.ID][2]

    def run(self, env_storage):
        '''
        This function takes the next ready tasks and run on the specific PE
        and update the env_storage.ready .list accordingly.
        '''
        
        DTPM_module = DTPM.DTPMmodule(self.args, self.env, env_storage, self.resource_matrix, self.PEs)

        for resource, current_PE in zip(self.resource_matrix.list, self.PEs):
            DTPM_policies.initialize_frequency(resource, current_PE)
            
        while True:  # Continue till the end of the simulation
            
            if self.env.now % self.sampling_rate == 0:
                # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                DTPM_module.evaluate_idle_PEs()
                
            if self.gym_env._get_done():
                if (self.scheduler_name == 'socrates') or (self.scheduler_name == 'scarl'):
                    cts = {ct.ID: ct for ct in env_storage.TaskQueues.completed.list}
                    self.task_completed += [(ti, cts[ti.task_id]) for ti in self.infos if ti.task_id in cts.keys()]
                    self.infos = [x for x in self.infos if x.task_id not in cts.keys()]            
                break
            
            if self.args.shared_memory:
                # this section is activated only if shared memory is used

                # Initialize $remove_from_wait_ready which will populate tasks
                # to be removed from the wait ready queue
                remove_from_wait_ready = []

                for i, waiting_task in enumerate(env_storage.TaskQueues.wait_ready.list):
                    if waiting_task.time_stamp <= self.env.now:
                        env_storage.TaskQueues.ready.list.append(waiting_task)
                        remove_from_wait_ready.append(waiting_task)
                # at the end of this loop, all the waiting tasks with a time stamp
                # equal or smaller than the simulation time will be added to
                # the ready queue list

                # Remove the tasks from wait ready queue that have been moved to ready queue
                for task in remove_from_wait_ready:
                    env_storage.TaskQueues.wait_ready.list.remove(task)

            if (self.scheduler_name == 'socrates') or (self.scheduler_name == 'scarl'):
                reward = self.gym_env._get_reward(env_storage)
                self.rewards_by_flops.append(reward)

            if self.scheduler_name == 'deepsocs':
                from gym_ds3.schedulers.deepsocs.preprocess_deepsocs import preprocess_deepsocs
                
                if len(env_storage.TaskQueues.ready.list) != 0:    
                    o = preprocess_deepsocs(self.gym_env)
                
                    action, buf_batch = self.scheduler.schedule(self.gym_env, o, 1, True)
                    
                    self.select_action(env_storage, action)
                
                    reward = self.gym_env._get_reward(env_storage)
                
                    if buf_batch is not None:
                        # valid action
                        len_batch = len(buf_batch['wall_time'])
                        buf_batch['reward'] = [reward for _ in range(len_batch)]
                        buf_batch['done'] = [False for _ in range(len_batch)]
                        env_storage.buf.store(buf_batch)
                    else:
                        # no action -> update reward and wall_time
                        self.gym_env.update_reward(self.env.now, reward)
                
                if action is not None:
                    # Update the execution queue based on task's info
                    self.update_execution_queue(env_storage, env_storage.TaskQueues.ready.list)
                
                    # update reward
                    reward = self.gym_env._get_reward(env_storage)
                    self.gym_env.update_reward(self.env.now, reward)
                
            else:
                if len(env_storage.TaskQueues.ready.list) != 0:
                    
                    ### Random ###
                    if self.scheduler_name == 'random':
                        for rt in env_storage.TaskQueues.ready.list:
                            rt.PE_ID = np.random.randint(self.gym_env.num_pes)
            
                    ### SCARL ###
                    elif self.scheduler_name == 'scarl':
                        from gym_ds3.schedulers.scarl.preprocess_scarl import preprocess_scarl

                        for rt in env_storage.TaskQueues.ready.list:
                            task_obs, pe_obs = preprocess_scarl(self.gym_env, rt.ID)
                            task_obs, pe_obs = torch.from_numpy(task_obs).type(torch.float), torch.from_numpy(pe_obs).type(torch.float)

                            res = self.scheduler.forward(task_obs, pe_obs)
                            selected_task, selected_pe, log_prob = res

                            ti = TaskInformation(
                                state_at_scheduling=(task_obs, pe_obs),
                                task_id=rt.ID,
                                task_base_id=rt.base_ID,
                                probs_idx=rt.ID % (self.scheduler.num_tasks_in_jobs * self.max_num_jobs),
                                action_at_scheduling=selected_pe,
                                timestep_of_scheduling=self.gym_env.now()
                            )

                            self.infos.append(ti)

                            rt.PE_ID = selected_pe
                            
                        cts = {ct.ID: ct for ct in env_storage.TaskQueues.completed.list}
                        self.task_completed += [(ti, cts[ti.task_id]) for ti in self.infos if ti.task_id in cts.keys()]
                        self.infos = [x for x in self.infos if x.task_id not in cts.keys()]
                    
                    ### SOCRATES ###        
                    elif self.scheduler_name == 'socrates':
                        from gym_ds3.schedulers.socrates.preprocess_socrates import preprocess_socrates

                        for rt in env_storage.TaskQueues.ready.list:
                            s_t = preprocess_socrates(self.gym_env)
                            s_t = torch.from_numpy(s_t).type(torch.float)

                            _, log_prob, _ = self.scheduler.forward(s_t)
                            pi = torch.exp(log_prob)

                            action_t = pi.multinomial(num_samples=1).detach()
                            action_at_scheduling = action_t[rt.ID % (self.scheduler.num_tasks_in_jobs * self.max_num_jobs)].item()

                            ti = TaskInformation(
                                state_at_scheduling=s_t,
                                task_id=rt.ID,
                                task_base_id=rt.base_ID,
                                probs_idx=rt.ID % (self.scheduler.num_tasks_in_jobs * self.max_num_jobs),
                                action_at_scheduling=action_at_scheduling,
                                timestep_of_scheduling=self.gym_env.now()
                            )

                            self.infos.append(ti)
                                
                            rt.PE_ID = action_at_scheduling

                        cts = {ct.ID: ct for ct in env_storage.TaskQueues.completed.list}
                        self.task_completed += [(ti, cts[ti.task_id]) for ti in self.infos if ti.task_id in cts.keys()]
                        self.infos = [x for x in self.infos if x.task_id not in cts.keys()]

                    ### HEURISTIC ###
                    else:
                        dicts = self.scheduler.schedule(env_storage)
                        self.select_action(env_storage, dicts)
                        
                    # Update the execution queue based on task's info
                    self.update_execution_queue(env_storage, env_storage.TaskQueues.ready.list)
                
            # Initialize $remove_from_executable which will populate tasks to be removed from the executable queue
            remove_from_executable = []

            # Go over each task in the executable queue
            if len(env_storage.TaskQueues.executable.list) != 0:
                for i, executable_task in enumerate(env_storage.TaskQueues.executable.list):
                    is_time_to_execute = (executable_task.time_stamp <= self.env.now)
                    PE_has_capacity = (len(self.PEs[executable_task.PE_ID].queue) < self.PEs[executable_task.PE_ID].capacity)
                    task_has_assignment = (executable_task.PE_ID != -1)

                    dynamic_dependencies_met = True

                    dependencies_completed = []
                    for dynamic_dependency in executable_task.dynamic_dependencies:
                        dependencies_completed = dependencies_completed + list(
                            filter(lambda completed_task: completed_task.ID == dynamic_dependency,
                                   env_storage.TaskQueues.completed.list))
                    if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                        dynamic_dependencies_met = False

                    if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met and task_has_assignment:
                        self.PEs[executable_task.PE_ID].queue.append(executable_task)

                        current_resource = self.resource_matrix.list[executable_task.PE_ID]
                        executable_task.wait_time = max(0, self.env.now - executable_task.ready_time)
                        
                        self.env.process(self.PEs[executable_task.PE_ID].execute(
                            self, executable_task, current_resource, env_storage, DTPM_module))

                        remove_from_executable.append(executable_task)

            # Remove the tasks from executable queue that have been executed by a resource
            for task in remove_from_executable:
                env_storage.TaskQueues.executable.list.remove(task)

            # The simulation tick is completed. Wait till the next interval
            yield self.env.timeout(self.args.simulation_clk)
