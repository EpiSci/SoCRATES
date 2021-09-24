'''
Description: This file contains the code for the job generator
'''
import copy
import random

import numpy as np
import simpy

from gym_ds3.envs.core.job_dag import JobDAG


class JobGenerator:
    """
    Define the JobGenerator class to handle dynamic job generation
    """
    def __init__(self, args, scale, gym_env, env, PEs, input_data, env_storage):
        """
        env: Pointer to the current simulation environment
        resource_matrix: The data structure that defines power/performance
            characteristics of the PEs for each supported task
        jobs: The list of all jobs given to DASH-Sim
        scheduler: Pointer to the DASH_scheduler
        PE_list: The PEs available in the current SoCs
        """
        self.args = args
        self.scale = scale
        self.gym_env = gym_env
        self.env = env
        self.jobs = input_data[0]
        self.resource_matrix = input_data[1]
        self.PEs = PEs     
        self.env_storage = env_storage
    
        self.generate_job = True                                                # Initially $generate_job is True so that as soon as run function is called
                                                                                #   it will start generating jobs
        self.generated_job_list = []                                            # List of all jobs that are generated
        self.offset = 0                                                         # This value will be used to assign correct ID numbers for incoming tasks

        if args.pss:
            self.start = False
        else:
            self.start = True

        self.capacity = self.args.max_num_jobs

        if len(self.jobs.list) == 4:
            self.job_probabilities = [0.25,0.25,0.25,0.25] # 4
        elif len(self.jobs.list) == 5:
            self.job_probabilities = [0.2,0.2,0.2,0.2,0.2] # 5
        elif len(self.jobs.list) == 6:
            self.job_probabilities = [0.17,0.17,0.17,0.17,0.17,0.15] # 6
        elif len(self.jobs.list) == 7:
            self.job_probabilities = [0.14,0.14,0.14,0.14,0.14,0.14,0.16] # 7
        elif len(self.jobs.list) == 8:
            self.job_probabilities = [0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.16] # 8
        elif len(self.jobs.list) == 9:
            self.job_probabilities = [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.12] # 9
        elif len(self.jobs.list) == 10:
            self.job_probabilities = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] # 10
        else:
            print('[E] Currently not support jobs more than 10.')
            raise Exception

    def get_graph_levels(self, adj_mat, num_nodes):
        level = 0

        for n in range(num_nodes):
            if sum(adj_mat[:, n]) != 0:
                level += 1

        return level

    def run(self):
        i = 0  # Initialize the iteration variable

        while self.generate_job:  # Continue generating jobs till #generate_job is False
            if self.env_storage.results.job_counter >= self.capacity and self.start:
                try:
                    yield self.env.timeout(self.args.simulation_clk)
                except simpy.exceptions.Interrupt:
                    pass
            else:
                num_of_apps = len(self.jobs.list)

                if not self.start:
                    while self.env_storage.results.job_counter < self.capacity:
                        selection = np.random.choice(list(range(num_of_apps)), 1, p=self.job_probabilities)
                        self.generated_job_list.append(copy.deepcopy(self.jobs.list[int(selection)]))
                        
                        for ii in range(len(self.generated_job_list[i].task_list)):
                            next_task = self.generated_job_list[i].task_list[ii]
                            next_task.jobID = i
                            next_task.base_ID = ii  # also record the original ID of the next task
                            next_task.ID = ii + self.offset  # and change the ID of the task accordingly

                            if next_task.head:
                                next_task.job_start = self.env.now

                            for k in range(len(next_task.predecessors)):
                                next_task.predecessors[k] += self.offset

                            if len(next_task.predecessors) > 0:
                                self.env_storage.TaskQueues.outstanding.list.append(next_task)
                            else:
                                next_task.ready_time = self.env.now
                                self.env_storage.TaskQueues.ready.list.append(next_task)

                        self.offset += len(self.generated_job_list[i].task_list)

                        job_dag = JobDAG(self.generated_job_list[i])
                        self.env_storage.job_dags.add(job_dag)
                        job_dag.start_inject_time = self.env.now

                        i += 1
                        self.env_storage.results.job_counter += 1

                    self.start = True

                else:
                    if self.args.verbose:
                        print('[D] Time %d: Job generator added job %d' % (self.env.now, i + 1))

                    if self.args.simulation_mode == 'validation':
                        self.env_storage.Validation.generated_jobs.append(i + 1)

                    selection = np.random.choice(list(range(num_of_apps)), 1, p=self.job_probabilities)
                    self.generated_job_list.append(copy.deepcopy(self.jobs.list[int(selection)]))

                    for ii in range(len(self.generated_job_list[i].task_list)):
                        next_task = self.generated_job_list[i].task_list[ii]
                        next_task.jobID = i

                        next_task.base_ID = ii  # also record the original ID of the next task
                        next_task.ID = ii + self.offset  # and change the ID of the task accordingly

                        if next_task.head:
                            # When a new job is generated, its execution is also started
                            next_task.job_start = self.env.now

                        for k in range(len(next_task.predecessors)):
                            # also change the predecessors of the newly added task, accordingly
                            next_task.predecessors[k] += self.offset

                        if len(next_task.predecessors) > 0:
                            self.env_storage.TaskQueues.outstanding.list.append(next_task)
                        else:
                            # Add the task to the ready queue since it has no predecessors
                            next_task.ready_time = self.env.now
                            self.env_storage.TaskQueues.ready.list.append(next_task)

                    self.offset += len(self.generated_job_list[i].task_list)

                    job_dag = JobDAG(self.generated_job_list[i])
                    self.env_storage.job_dags.add(job_dag)
                    job_dag.start_inject_time = self.env.now

                    # Update the job ID
                    i += 1
                    self.env_storage.results.job_counter += 1

                    # if running on validation mode
                    if self.args.simulation_mode == 'validation':
                        # check if max number of jobs, given in config file, are created
                        VAL_N_JOBS = 1
                        if i >= VAL_N_JOBS:
                            self.generate_job = False  # if yes, no more jobs will be added to simulation

            # assign an exponentially distributed random variable to $wait_time
            self.wait_time = int(random.expovariate(1 / self.scale))
            
            # new job addition will be after this wait time
            yield self.env.timeout(self.wait_time)
