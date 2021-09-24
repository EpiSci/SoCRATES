import numpy as np

from gym_ds3.envs.utils.helper_dict import OrderedSet


class ReplayBuffer:

    def __init__(self):
        self.exp = {
            'obs_change': [],
            'node_inputs': [],
            'pe_feature': [],
            'comm_feature': [],
            'gcn_mats': [],
            'gcn_masks': [],
            'summ_mats': [],
            'running_dag_mat': [],
            'dag_summ_backward': [],
            'dag_summ_back_mat': [],
            'node_act_vec': [],
            'node_valid_mask': [],
            'pe_valid_mask': [],
            'job_state_change': [],
            'wall_time': [],
            'a1': [],
            'a2': [],
            'reward': [],
            'done': [],
        }

    def store(self, buf_batch):
        for key, val in buf_batch.items():
            self.exp[key].extend(val)

    def get(self):
        return self.exp

    def reset(self):
        self.exp = {
            'obs_change': [],
            'node_inputs': [],
            'pe_feature': [],
            'comm_feature': [],
            'gcn_mats': [],
            'gcn_masks': [],
            'summ_mats': [],
            'running_dag_mat': [],
            'dag_summ_backward': [],
            'dag_summ_back_mat': [],
            'node_act_vec': [],
            'node_valid_mask': [],
            'pe_valid_mask': [],
            'job_state_change': [],
            'wall_time': [],
            'a1': [],
            'a2': [],
            'reward': [],
            'done': []
        }


class EnvStorage(object):
    def __init__(self):
        self.job_dags = OrderedSet()
        self.completed_job_dags = OrderedSet()

        self.buf = ReplayBuffer()
        
    class PerfStatics:
        '''
        Define the PerfStatics class to calculate power consumption and total execution time.
        '''
        def __init__(self):
            self.execution_time = 0.0                   # The total execution time (us)
            self.energy_consumption = 0.0               # The total energy consumption (uJ)
            self.cumulative_exe_time = 0.0              # Sum of the execution time of completed tasks (us)
            self.cumulative_energy_consumption = 0.0    # Sum of the energy consumption of completed tasks (us)
            self.injected_jobs = 0                      # The number of jobs entered the system (i.e. the ready queue)
            self.completed_jobs = 0                     # Count the number of jobs that are completed
            self.ave_execution_time = 0.0               # Average execution time for the jobs that are finished
            self.job_counter = 0
            self.job_counter_list = []
            self.sampling_rate_list = []

            # more

    # Instantiate the object that will store the performance statistics
    # global results
    class Validation:
        """
        Define the Validation class to compare the generated and completed jobs
        """
        start_times = []
        finish_times = []
        generated_jobs = []
        injected_jobs = []
        completed_jobs = []

    class ResourceManager:
        """
        Define the ResourceManager class to maintain
        the list of the resource in our DASH-SoC model.
        """
        def __init__(self):
            self.list = []                          # list of available resources
            self.comm_band = []                     # This variable represents the communication bandwidth matrix


    class TaskManager:
        """
        Define the TaskManager class to maintain the
        list of the tasks in our DASH-SoC model.
        """
        def __init__(self):
            self.list = []                          # List of available tasks

    class ApplicationManager:
        """
        Define the ApplicationManager class to maintain the
        list of the applications (jobs) in our DASH-SoC model.
        """
        def __init__(self):
            self.list = []                          # List of all applications

    class TaskQueues:
        '''
        Define the TaskQueues class to maintain the
        all task queue lists
        '''
        def __init__(self):
            self.outstanding = []                   # List of *all* tasks waiting to be processed
            self.ready = []                         # List of tasks that are ready for processing
            self.running = []                       # List of currently running tasks
            self.completed = []                     # List of completed tasks
            self.wait_ready = []                    # List of task waiting for being pushed into ready queue
            self.executable = []                    # List of task waiting for being executed


class Resource:
    """
    Define the Resource class to define a resource
    It stores properties of the resources.
    """
    def __init__(self):
        self.type = ''                       # The type of the resource (CPU, FFT_ACC, etc.)
        self.name = ''                       # Name of the resource
        self.ID = -1                         # The unique ID of the resource. "-1" if it is not initialized
        self.capacity = 1                    # Resource capacity. Default value is 1.
        self.num_of_functionalities = 0      # This variable shows how many different task this resource can run
        self.supported_functionalities = []  # List of all tasks can be executed by Resource
        self.performance = []                # List of runtime (in micro seconds) for each supported task
        self.power_consumption = []          # list of power consumption (in milliwatts) for each supported task
        self.idle = True                     # initial state of Resource which idle and ready for a task
        self.DVFS = ''                       # DVFS mode
        self.OPP = []                        # List of all Operating Performance Points (OPPs)
                                             # each OPP is a <frequency, voltage> tuple.

class Applications:
    """
    Define the Applications class to maintain the
    all information about an application (job)
    """
    def __init__(self):
        self.name = ''                          # The name of the application
        self.task_list = []                     # List of all tasks in an application
        self.comm_vol = []                      # This variable represents the communication volume matrix


class Tasks:
    """
    Define the Tasks class to maintain the list
    of tasks. It stores properties of the tasks.
    """
    def __init__(self):
        self.name = ''  # The name of the task
        self.jobname = ''  # This task belongs to job with this name
        self.ID = -1  # This is the unique ID of the task. "-1" means it is not initialized
        self.base_ID = -1  # This ID will be used to calculate the data volume from one task to another
        self.jobID = -1  # This task belongs to job with this ID
        self.PE_ID = -1  # Holds the PE ID on which the task will be executed

        self.predecessors = []  # List of all task IDs to identify task dependency
        self._predecessors = []  # List of predecessors (not deleted)
        self.predecessors_finish_times = []  # List of all task IDs to identify task dependency
        self.predecessor_PE_IDs = []  # List of all task IDs to identify task dependency
        self.est = -1  # This variable represents the earliest time that a task can start
        self.deadline = -1  # This variable represents the deadline for a task
        self.head = False  # If head is true, this task is the leading (the first) element in a task graph
        self.tail = False  # If tail is true, this task is the end (the last) element in a task graph

        self.start_time = -1  # Execution start time of a task
        self.finish_time = np.inf  # Execution finish time of a task
        self.job_start = -1  # Holds the execution start time of a head task (also execution start time for a job)
        self.ready_time = -1
        self.duration = -1
        self.wait_time = -1
        self.task_elapsed_time_max_freq = 0  # Indicate the current execution time for a given task (used by the checkpoint mechanism)

        self.order = -1  # Relative ordering of this task on a particular PE

        self.dynamic_dependencies = []  # List of dynamic dependencies that a scheduler requests are satisfied before task launch
        self.ready_wait_times = []  # List holding wait times for a task for being ready due to communication time from its predecessor
        self.execution_wait_times = []  # List holding wait times for a task for being execution-ready due to communication time between memory and a PE
        self.PE_to_PE_wait_time = []  # List holding wait times for a task for being execution-ready due to PE to PE communication time

        self.time_stamp = -1  # This values used to check whether all data for the task is transferred or not
        self.slack = 100000  # How many us can a task be delayed
        self.min_freq = -1  # Minimum frequency of the PE the task was assigned to
        self.max_freq = -1  # Maximum frequency of the PE the task was assigned to
        self.avg_freq = -1  # Average frequency of the PE the task was assigned to
        self.sum_freq = []
        self.edp = -1  # Edp of a task
