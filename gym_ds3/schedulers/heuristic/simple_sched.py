import copy

import numpy as np


class MET(object):
    """
    MET returns the ID of the resource that executes the current task with the minimum time
    """

    def __init__(self, env):
        self.env = env

    def schedule(self, state):
        """
        :param env_storage: data stored in environment
        :return scheduled_tasks: Scheduled tasks and their resources
        """

        scheduled_tasks = {}
        env_storage = self.env.env_storage

        # Initialize a list to record number of assigned tasks to a PE
        # for every scheduling instance

        # Initialize a list
        assigned = [0] * (len(self.env.pes))

        # For all tasks in the ready queue, find the resource with minimum execution time of the task
        for idx, task in enumerate(env_storage.TaskQueues.ready.list):

            exec_times = [np.inf] * (len(self.env.pes)) # initialize execution time for all resources to infinite

            for i in range(len(self.env.resource_matrix_list)):
                if task.name in self.env.resource_matrix_list[i].supported_functionalities:
                    ind = self.env.resource_matrix_list[i].supported_functionalities.index(task.name)
                    exec_times[i] = self.env.resource_matrix_list[i].performance[ind]

            min_of_exec_times = min(exec_times)  # the minimum execution time of the task among PEs
            count_minimum = exec_times.count(min_of_exec_times)  # check if there are multiple resources with minimum execution time

            # if there are two or more PEs satisfying minimum execution
            # then we should try to utilize all those PEs
            if count_minimum > 1:

                # if there are tow or more PEs satisfying minimum execution
                # populate the IDs of those PEs into a list
                min_PE_IDs = [i for i, x in enumerate(exec_times) if x == min_of_exec_times]

                # then check whether those PEs are busy or idle
                PE_check_list = [True if not self.env.pes[index].idle else False for i, index in enumerate(min_PE_IDs)]

                # assign tasks to the idle PEs instead of the ones that are currently busy
                if (True in PE_check_list) and (False in PE_check_list):
                    for PE in PE_check_list:
                        # if a PE is currently busy remove that PE from $min_PE_IDs list
                        # to schedule the task to a idle PE
                        if PE:
                            min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])

                # then compare the number of the assigned tasks to remaining PEs
                # and choose the one with the lowest number of assigned tasks
                assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                scheduled_tasks[task.ID] = (min_PE_IDs[PE_ID_index], idx, [])

            else:
                scheduled_tasks[task.ID] = (exec_times.index(min_of_exec_times), idx, [])
            # end of if count_minimum >1:
            # since one task is just assigned to a PE, increase the number by 1
            assigned[task.PE_ID] += 1

            # end of if task.PE_ID == -1:
            # end of for task in list_of_ready:
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        return scheduled_tasks


class EFT(object):

    def __init__(self, env):
        super(EFT, self).__init__(env)
        self.args = env.args

    def schedule(self, state):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task
        '''
        scheduled_tasks = {}
        env_storage = self.env.env_storage

        for idx, task in enumerate(env_storage.TaskQueues.ready.list):

            comparison = [np.inf] * len(self.env.pes)  # Initialize the comparison vector
            comm_ready = [0] * len(self.env.pes)  # A list to store the max communication times for each PE

            if self.args.verbose:
                print('[D] Time %s: The scheduler function is called with task %s'
                      % (self.env.now(), task.ID))

            for i in range(len(self.env.resource_matrix_list)):
                # if the task is supported by the resource, retrieve the index of the task
                if task.name in self.env.resource_matrix_list[i].supported_functionalities:
                    ind = self.env.resource_matrix_list[i].supported_functionalities.index(task.name)

                    # $PE_comm_wait_times is a list to store the estimated communication time
                    # (or the remaining communication time) of all predecessors of a task for a PE
                    # As simulation forwards, relevant data is being sent after a task is completed
                    # based on the time instance, one should consider either whole communication
                    # time or the remaining communication time for scheduling
                    PE_comm_wait_times = []

                    # $PE_wait_time is a list to store the estimated wait times for a PE
                    # till that PE is available if the PE is currently running a task
                    PE_wait_time = []

                    job_ID = -1  # Initialize the job ID

                    # Retrieve the job ID which the current task belongs to
                    for ii, job in enumerate(self.env.jobs.list):
                        if job.name == task.jobname:
                            job_ID = ii

                    for predecessor in self.env.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                        # data required from the predecessor for $ready_task
                        c_vol = self.env.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + task.ID - task.base_ID

                        # Initialize following two variables which will be used if
                        # PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1

                        for completed in env_storage.TaskQueues.completed.list:
                            if completed.ID == real_predecessor_ID:
                                predecessor_PE_ID = completed.PE_ID
                                predecessor_finish_time = completed.finish_time

                        if self.args.shared_memory:
                            # Compute the communication time considering the shared memory
                            # only consider memory to PE communication time
                            # since the task passed the 1st phase (PE to memory communication)
                            # and its status changed to ready

                            memory_to_PE_band = self.env.resource_matrix.comm_band[self.env.resource_matrix_list[-1].ID, i]
                            shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                            PE_comm_wait_times.append(shared_memory_comm_time)
                            if self.args.verbose:
                                print('[D] Time %s: Estimated communication time between '
                                      'memory to PE %s from task %s to task %s is %d'
                                      % (self.env.now(), i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                        # PE_to_PE
                        else:
                            # Compute the PE to PE communication time
                            PE_to_PE_band = self.env.resource_matrix.comm_band[predecessor_PE_ID, i]
                            PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                            PE_comm_wait_times.append(max((predecessor_finish_time +
                                                           PE_to_PE_comm_time - self.env.now()), 0))

                            if self.args.verbose:
                                print('[D] Time %s: Estimated communication time between PE %s to PE'
                                      ' %s from task %s to task %s is %d'
                                      % (self.env.now(), predecessor_PE_ID, i, real_predecessor_ID,
                                         task.ID, PE_comm_wait_times[-1]))

                        
                        # $comm_ready contains the estimated communication time
                        # for the resource in consideration for scheduling
                        # maximum value is chosen since it represents the time required for all
                        # data becomes available for the resource.
                        comm_ready[i] = (max(PE_comm_wait_times))
                    # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                    # if a resource currently is executing a task, then the estimated remaining time
                    # for the task completion should be considered during scheduling
                    PE_wait_time.append(max((self.env.pes[i].available_time - self.env.now()), 0))

                    # update the comparison vector accordingly
                    comparison[i] = self.env.resource_matrix_list[i].performance[ind] \
                                    + max(comm_ready[i], PE_wait_time[-1])
                # end of if (task.name in...
            # end of for i in range(len(self.env.resource_matrix_list)):

            # after going over each resource, choose the one which gives the minimum result
            task_PE_ID = comparison.index(min(comparison))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.env.pes[task_PE_ID].available_time = self.env.now() + comparison[task_PE_ID]
            scheduled_tasks[task.ID] = (task_PE_ID, idx, [])

        return scheduled_tasks


class ETF(object):

    def __init__(self, env):
        super(ETF, self).__init__(env)
        self.args = self.env.args

    def schedule(self, state):
        env_storage = self.env.env_storage
        ready_list = copy.deepcopy(env_storage.TaskQueues.ready.list)

        task_counter = 0

        scheduled_tasks = {}

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1
            shortest_comparison = [np.inf] * len(self.env.pes)

            for task in ready_list:

                comparison = [np.inf] * len(self.env.pes)  # Initialize the comparison vector
                comm_ready = [0] * len(self.env.pes)  # A list to store the max communication times for each PE

                if self.args.verbose:
                    print('[D] Time %s: The scheduler function is called with task %s'
                          % (self.env.now(), task.ID))

                for i in range(len(self.env.resource_matrix_list)):
                    # if the task is supported by the resource, retrieve the index of the task
                    if task.name in self.env.resource_matrix_list[i].supported_functionalities:
                        ind = self.env.resource_matrix_list[i].supported_functionalities.index(task.name)

                        # $PE_comm_wait_times is a list to store the estimated communication time
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []

                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []

                        job_ID = -1  # Initialize the job ID

                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.env.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii

                        for predecessor in self.env.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.env.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID

                            # Initialize following two variables which will be used if
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1

                            for completed in env_storage.TaskQueues.completed.list:
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    break
                            
                            if self.args.shared_memory:
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready

                                memory_to_PE_band = self.env.resource_matrix.comm_band[
                                    self.env.resource_matrix_list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if self.args.verbose:
                                    print('[D] Time %s: Estimated communication time between '
                                        'memory to PE-%s from task %s to task %s is %d'
                                        % (self.env.now(), i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            # PE_to_PE
                            else:  
                                # Compute the PE to PE communication time
                                PE_to_PE_band = self.env.resource_matrix.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                PE_comm_wait_times.append(
                                    max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now()), 0))

                                if self.args.verbose:
                                    print('[D] Time %s: Estimated communication time between PE-%s to PE-%s '
                                        'from task %s to task %s is %d'
                                        % (self.env.now(), predecessor_PE_ID, i, real_predecessor_ID, task.ID,
                                           PE_comm_wait_times[-1]))

                            # $comm_ready contains the estimated communication time
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource.
                            comm_ready[i] = (max(PE_comm_wait_times))
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.env.pes[i].available_time - self.env.now()), 0))

                        # update the comparison vector accordingly
                        comparison[i] = self.env.resource_matrix_list[i].performance[ind] + max(comm_ready[i],
                                                                                              PE_wait_time[-1])

                        # after going over each resource, choose the one which gives the minimum result
                        resource_id = comparison.index(min(comparison))

                        # obtain the task ID, resource for the task with earliest finish time
                        # based on the computation
                        if min(comparison) < shortest_task_exec_time:
                            shortest_task_exec_time = min(comparison)
                            shortest_task_pe_id = resource_id
                            shortest_task = task
                            shortest_comparison = comparison

                    # end of if (task.name in...
                # end of for i in range(len(self.env.resource_matrix_list)):

            # assign PE ID of the shortest task
            index = [i for i, x in enumerate(env_storage.TaskQueues.ready.list) if x.ID == shortest_task.ID][0]

            env_storage.TaskQueues.ready.list[index], env_storage.TaskQueues.ready.list[task_counter] \
                = env_storage.TaskQueues.ready.list[task_counter], env_storage.TaskQueues.ready.list[index]

            scheduled_tasks[shortest_task.ID] = (shortest_task_pe_id, index, [])
            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.env.pes[shortest_task_pe_id].available_time = self.env.now() + \
                                                               shortest_comparison[shortest_task_pe_id]

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

            task_counter += 1

        return scheduled_tasks
    # end of ETF( list_of_ready)


class STF(object):

    def __init__(self, env):
        super(STF, self).__init__(env)

    def schedule(self, state):
        
        env_storage = self.env.env_storage
        ready_list = copy.deepcopy(env_storage.TaskQueues.ready.list)
        scheduled_tasks = {}

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1

            for task_id, task in enumerate(ready_list):

                min_time = np.inf  # Initialize the best performance found so far as a large number

                for i in range(len(self.env.resource_matrix_list)):
                    if task.name in self.env.resource_matrix_list[i].supported_functionalities:
                        ind = self.env.resource_matrix_list[i].supported_functionalities.index(task.name)

                        if self.env.resource_matrix_list[i].performance[ind] < min_time:
                            # Found resource with smaller execution time
                            min_time = self.env.resource_matrix_list[i].performance[ind]
                            # Update the best time found so far
                            resource_id = self.env.resource_matrix_list[i].ID  # Record the ID of the resource

                # Obtain the ID and resource for the shortest task in the current iteration
                if min_time < shortest_task_exec_time:
                    shortest_task_exec_time = min_time
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                # end of if (min_time < shortest_task_exec_time)

            # end of for task in list_of_ready:
            # At the end of this loop, we should have the minimum execution time
            # of a task across all resources

            # Assign PE ID of the shortest task
            index = [x.ID for i, x in enumerate(env_storage.TaskQueues.ready.list) if x.ID == shortest_task.ID][0]
            
            scheduled_tasks[index] = (shortest_task_pe_id, index, [])
            shortest_task.PE_ID = shortest_task_pe_id

            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

        return scheduled_tasks

        # end of for task in list_of_ready:
        # At the end of this loop, all ready tasks are assigned to the resources
        # on which the execution times are minimum. The tasks will execute
        # in the order of increasing execution times
