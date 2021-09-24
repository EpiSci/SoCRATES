'''
Description: This file contains the process elements and their attributes.
'''
import random
import simpy

from gym_ds3.envs.power import DTPM_power_models
from gym_ds3.envs.power import DTPM_policies


class PE:
    """
    A processing element (PE) is the basic resource that defines
    the simpy processes.
    A PE has a *name*, *utilization (float)* and a process (resource)
    """

    def __init__(self, env, gym_env, type, name, ID, capacity, std, agent=None):
        """
        env: Pointer to the current simulation environment
        name: Name of the current processing element
        ID: ID of the current processing element
        capacity: Number tasks that a resource can run simultaneously
        """
        self.env = env
        self.gym_env = gym_env
        self.args = self.gym_env.args

        self.type = type
        self.name = name
        self.ID = ID
        self.capacity = capacity
        self.std = std
        self.agent = agent
        
        self.task = None

        self.current_frequency = 0                                              # Indicate the current PE frequency
        self.current_voltage = 0                                                # Indicate the current PE voltage
        self.current_temperature_vector = [[self.args.xu3_t_ambient],           # Indicate the current PE temperature for each hotspot
                                           [self.args.xu3_t_ambient],
                                           [self.args.xu3_t_ambient],
                                           [self.args.xu3_t_ambient],
                                           [self.args.xu3_t_ambient]]
        self.current_power = 0                                                  # Indicate the current power dissipation (dynamic + static)
        self.current_leakage = 0                                                # Indicate the current leakage power
        self.energy_sampling_period = 0                                         # Indicate the energy consumption during the current sampling period
        self.total_energy = 0                                                   # Indicate the total energy consumed by the given PE

        self.Cdyn_alpha = 0                                                     # Variable that stores the dynamic capacitance * switching activity for each PE

        self.queue = []                                                         # List of currently running task on a PE
        self.available_time = 0                                                 # Estimated available time of PE
        self.available_time_list = [0] * self.capacity                          # Estimated available time for each core os the PE
        self.idle = True                                                        # The variable indicates whether the PE is active or not
        self.busy_dur = 0.

        self.info = []                                                          # List to record all the events happened on a PE
        self.process = simpy.Resource(env, capacity=self.capacity)
        self.task_start_times = []

        self.task_runtime = 0

        self.task_expected_total_time = 0

        if self.args.verbose:
            print('[D] Constructed PE-%d with name %s' % (ID, name))

    # Start the "run" process for this PE
    def execute(self, sim_manager, task, resource, env_storage, DVFS_module):
        """
        Run this PE to execute a given task.
        The execution time is retrieved from resource_matrix and task name
        """

        try:
            # Requesting the resource for the task
            with self.process.request() as req:
                yield req

                if self.current_frequency == 0:                                 # Initialize the frequency if it was not set yet
                    # Depending on the DVFS policy on this PE, set the initial frequency and voltage accordingly
                    if resource.DVFS != 'none' or len(resource.OPP) != 0:
                        DTPM_policies.initialize_frequency(resource, self)

                        # DASH_Sim_utils.f_trace_frequency(self.env.now, self)

                self.idle = False  # PE is not idle when the PE executes a task
                env_storage.TaskQueues.running.list.append(task)
                task.start_time = self.env.now  # Start time of task execution time
                self.task = task

                # if this is the leading task of this job, increment the injection counter
                if task.head and (self.env.now >= self.args.warmup_period):
                    env_storage.results.injected_jobs += 1
                    if self.args.verbose:
                        print('[D] Time %d: Total injected jobs becomes: %d'
                              % (self.env.now, env_storage.results.injected_jobs))

                    # Store the injected job for validation
                    if self.args.simulation_mode == 'validation':
                        env_storage.Validation.injected_jobs.append(task.jobID)

                    for jd in env_storage.job_dags:
                        if task.jobID == jd.jobID:
                            jd.is_running = True
                            jd.start_exec_time = self.env.now

                if self.args.verbose:
                    print('[D] Time %d: Task %s execution is started by PE-%d %s'
                          % (self.env.now, task.ID, self.ID, self.name))

                self.task_start_times.append(self.env.now)
                task.min_freq = self.current_frequency
                task.max_freq = self.current_frequency
                task.avg_freq = self.current_frequency
                task.sum_freq.append(self.current_frequency)

                # Retrieve the execution time from the model
                task_runtime_max_freq = get_execution_time(task, resource, self.std)
                self.task_expected_total_time = task_runtime_max_freq  # deepsocs: record total task time

                power_consumption_max_freq = DTPM_power_models.get_power_consumption_max_frequency(task, resource)
                # Compute the leakage power
                self.current_leakage = DTPM_power_models.compute_static_power_dissipation(self.args, self)
                # Based on the total power consumption and the leakage, get the dynamic power
                dynamic_power_max_freq = power_consumption_max_freq  -  self.current_leakage
                # Compute the capacitance and alpha based on the dynamic power
                self.Cdyn_alpha = DTPM_power_models.compute_Cdyn_and_alpha(resource, dynamic_power_max_freq)

                dynamic_energy = 0
                static_energy = 0
                task_runtime = 0
                task_complete = False
                task_elapsed_time = task.task_elapsed_time_max_freq

                while task_complete == False:
                    # The predicted time takes into account the current frequency
                    # and subtracts the time that the task already executed
                    predicted_exec_time = (task_runtime_max_freq - task_elapsed_time) + (task_runtime_max_freq - task_elapsed_time) * \
                        DTPM_power_models.compute_DVFS_performance_slowdown(resource, self.current_frequency)
                    window_remaining_time = self.args.sampling_rate - self.env.now % self.args.sampling_rate
                    # Test if the task finished before the next sampling period
                    if predicted_exec_time - window_remaining_time > 0:
                        # Run until the next sampling timestamp
                        simulation_step = window_remaining_time
                        slowdown = DTPM_power_models.compute_DVFS_performance_slowdown(resource, self.current_frequency) + 1
                        task_elapsed_time += window_remaining_time / slowdown
                    else:
                        # Run until the task ends
                        simulation_step = predicted_exec_time
                        task_complete = True

                    task_runtime += simulation_step
                    self.task_runtime = task_runtime

                    # Compute the dynamic energy
                    dynamic_power = DTPM_power_models.compute_dynamic_power_dissipation(self.current_frequency, self.current_voltage, self.Cdyn_alpha)
                    dynamic_energy +=  dynamic_power * simulation_step
                    # Compute the static energy
                    self.current_leakage = DTPM_power_models.compute_static_power_dissipation(self.args, self)
                    static_energy += self.current_leakage * simulation_step
                    self.current_power = dynamic_power + self.current_leakage
                    self.energy_sampling_period += (dynamic_power + self.current_leakage) * simulation_step
                    self.total_energy += dynamic_power * simulation_step + self.current_leakage * simulation_step

                    yield self.env.timeout(simulation_step)

                    task.task_elapsed_time_max_freq = task_elapsed_time

                    if self.env.now % self.args.sampling_rate == 0:
                        # Case 1: If the task is not complete, evaluate this PE at this moment
                        if task_complete == False:
                            DVFS_module.evaluate_PE(resource, self, self.env.now)

                # task completion
                task.finish_time = int(self.env.now)
                task.avg_freq = int(sum(task.sum_freq)/len(task.sum_freq))

                # As the task finished its execution, reset the task time
                task.task_elapsed_time_max_freq = 0

                if self.args.verbose:
                    print('[D] Time %d: Task %s execution is finished by PE-%d %s'
                          % (self.env.now, task.ID, self.ID, self.name))

                task_time = task.finish_time - task.start_time
                self.busy_dur += task_time

                task.duration = task_time
                self.idle = True

                # If there are no OPPs in the model, use the measured power consumption from the model
                if len(resource.OPP) == 0:
                    total_energy_task = dynamic_power_max_freq * task_time
                else:
                    total_energy_task = dynamic_energy + static_energy

                if task.tail:
                    env_storage.results.job_counter -= 1

                    if not self.args.simulation_mode == 'validation':
                        sim_manager.update_completed_queue(env_storage)

                    if self.env.now >= self.args.warmup_period:
                        env_storage.results.execution_time = self.env.now
                        env_storage.results.completed_jobs += 1

                        for completed in env_storage.TaskQueues.completed.list:
                            if (completed.head is True) and (completed.jobID == task.jobID):
                                env_storage.results.cumulative_exe_time += (self.env.now - completed.job_start)

                        for jd in env_storage.job_dags:
                            if task.jobID == jd.jobID:
                                jd.completion_time = self.env.now
                                jd.is_completed = True
                                jd.is_running = False
                                env_storage.completed_job_dags.add(jd)

                    # Store the completed job for validation
                    if self.args.simulation_mode == 'validation':
                        env_storage.Validation.completed_jobs.append(task.jobID)

                if self.args.verbose:
                    print('[I] Time %d: Task %s is finished by PE-%d %s with %.2f us'
                          % (self.env.now, task.ID, self.ID, self.name, round(task_time, 2)))
                
                task.edp = task_time * total_energy_task

                if self.env.now >= self.args.warmup_period:
                    env_storage.results.cumulative_energy_consumption += total_energy_task

                # Since the current task is processed, it should be removed
                # from the outstanding task queue
                sim_manager.update_ready_queue(env_storage, task)

                # Case 2: Evaluate the PE after the queues are updated
                if self.env.now % self.args.sampling_rate == 0:
                    DVFS_module.evaluate_PE(resource, self, self.env.now)

        except simpy.Interrupt:
            print('Expect an interrupt at %s' % self.env.now)

    def get_idle_rate(self):
        if self.env.now == 0:
            return 0.
        else:
            return 1. - (self.busy_dur / self.env.now)


def get_execution_time(task, resource, std):
    """
    returns the execution time of the current task.
    """
    task_ind = resource.supported_functionalities.index(task.name)  # Retrieve the index of the task
    execution_time = resource.performance[task_ind]  # Retrieve the mean execution time of a task
    if resource.performance[task_ind]:
        # Randomize the execution time based on a gaussian distribution
        randomized_execution_time = max(round(
            random.gauss(execution_time, std * execution_time)), 1)
        return randomized_execution_time
    else:  # if the expected execution time is 0, ie. if it is dummy task, then no randomization
        # If a task has a 0 us of execution (dummy ending task), it should stay the same
        return execution_time
    