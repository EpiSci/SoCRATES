import copy

import numpy as np


### SOCRATES ###
MAX_TIMESTAMP = int(2 ** 17)  # 131072
MAX_NUM_TASKS = int(2 ** 14)  # 16384
MAX_COMP_TIME = int(2 ** 5)   # 32
MAX_WAIT_TIME = int(2 ** 10)  # 1024
NUM_RESOURCE_MATRIX = 3
NUM_STATUSES = 6
NUM_PREDECESSORS = 30


def preprocess_socrates(gym_env):
    job_dags = [jd for jd in gym_env.env_storage.job_dags if not jd.is_completed]

    if gym_env.updated and gym_env.env_storage.TaskQueues.ready.list:
        gym_env.updated = False

    _, num_remained_dependency = get_num_dependency(gym_env.env_storage)
    obs = floating_preprocess(gym_env, job_dags, num_remained_dependency)
    
    return obs


def dependency_len(dep_graph, cur_task_idx):
    if sum(dep_graph[cur_task_idx, :]) == 0:  # No next task
        return 0
    else:
        lengths = []
        for next_task_idx in range(len(dep_graph[cur_task_idx, :])):
            if dep_graph[cur_task_idx, next_task_idx] != 0:
                lengths.append(dependency_len(dep_graph, next_task_idx))
        return max(lengths) + 1


def get_num_dependency(env_storage):
    num_remained_dependency = {}  # {jobID: remained_dependency_length, ...}
    job_adj_matrix = {}  # {jobID: adj_matrix, ...}
    job_task_numbers = {}  # {jobID: task_num, ...}
    job_task_cumulative_numbers = {-1: 0}  # {jobID: sum of previous task_num, ...}
    for job_dag in env_storage.job_dags:
        job_adj_matrix[job_dag.jobID] = copy.deepcopy(np.zeros_like(np.zeros((len(job_dag.tasks), len(job_dag.tasks)))))
        job_task_numbers[job_dag.jobID] = len(job_dag.tasks)
        job_task_cumulative_numbers[job_dag.jobID] = \
            len(job_dag.tasks) + job_task_cumulative_numbers[job_dag.jobID - 1]

    all_tasks = env_storage.TaskQueues.outstanding.list + env_storage.TaskQueues.ready.list
    for task in all_tasks:
        cur_task_predecessor = task.predecessors
        cur_task_predecessor = [t - job_task_cumulative_numbers[task.jobID - 1] for t in cur_task_predecessor]
        for p in cur_task_predecessor:
            try:
                job_adj_matrix[task.jobID][task.ID - job_task_cumulative_numbers[task.jobID - 1], p] = 1
            except:
                # e.g. 1 30 18 [1, 18] {-1: 0, 0: 15, 1: 31, 2: 46}
                print(task.jobID, task.ID, task.predecessors, p, cur_task_predecessor, job_task_cumulative_numbers)

    for jobID in job_adj_matrix.keys():
        adj = job_adj_matrix[jobID]
        task_num = job_task_numbers[jobID]
        remained_dependency_length = dependency_len(adj, task_num - 1)
        num_remained_dependency[jobID] = remained_dependency_length

    for job_dag in env_storage.job_dags:
        if job_dag.is_completed:
            num_remained_dependency.pop(job_dag.jobID)

    return np.sum(list(num_remained_dependency.values())), num_remained_dependency


def get_repr_size():
    return int(np.log2(MAX_WAIT_TIME)) + int(np.log2(MAX_WAIT_TIME)) + \
           NUM_PREDECESSORS + int(np.log2(MAX_TIMESTAMP)) + NUM_STATUSES


def get_binary_repr(num, max_num, allow_undefined=False):
    if num == -1:  # Values take -1 in certain cases.
        if allow_undefined:
            return np.array(
                [int(i) for i in np.binary_repr(0, width=int(np.log2(max_num)))],
                dtype=np.float32)
        else:
            raise Exception("Got undefined value!")
    if num < max_num:
        return np.array(
            [int(i) for i in np.binary_repr(num, width=int(np.log2(max_num)))],
            dtype=np.float32)
    elif num == np.inf:
        return np.array(
            [int(i) for i in np.binary_repr(0, width=int(np.log2(max_num)))],
            dtype=np.float32)
    else:
        print(num)
        raise Exception("Expected num < max_num")


def get_one_hot_repr(num, max_num):
    return np.eye(max_num, dtype=np.float32)[num]


def get_predecessor_repr(task, env_storage):
    predecessor_repr = np.zeros((NUM_PREDECESSORS,), dtype=np.float)

    if len(task.predecessors) == 0:
        pass
    else:
        predecessors = task.predecessors
        for pred in predecessors:
            if pred in [t.ID for t in env_storage.TaskQueues.completed.list]:
                continue
            else:
                if task.jobID < 1:
                    predecessor_repr[pred] = 1
                else:
                    predecessor_repr[pred % task.jobID] = 1

    return predecessor_repr


def floating_preprocess(gym_env, job_dags, num_remained_dependency):
    EPS = float(1e-5)
    
    args = gym_env.args
    env_storage = gym_env.env_storage
    curr_time = gym_env.now()

    num_children_tasks = np.asarray(
        [(len(env_storage.TaskQueues.outstanding.list) + len(env_storage.TaskQueues.ready.list)) / \
         (args.max_num_jobs * args.num_tasks_in_job)],
        dtype=np.float)

    # normalized num_remained_dependency
    b_level = np.zeros((args.max_num_jobs), dtype=np.float)
    for id, jd in enumerate([jd for jd in env_storage.job_dags if not jd.is_completed]):
        if jd.job.name == 'Top_1':
            b_level[id] = num_remained_dependency[jd.jobID] / 4
        else:
            b_level[id] = num_remained_dependency[jd.jobID] / 5

    job_wait_time = np.zeros((args.max_num_jobs), dtype=np.float)
    for jid, job_dag in enumerate(job_dags):
        if job_dag.start_exec_time is not np.inf:
            job_wait_time[jid] = \
                ((job_dag.start_exec_time - job_dag.start_inject_time) / (job_dag.start_exec_time + EPS))

    wait_task = env_storage.TaskQueues.ready.list + env_storage.TaskQueues.executable.list
    running = env_storage.TaskQueues.running.list
    outstanding = env_storage.TaskQueues.outstanding.list

    task_processed = np.zeros((args.max_num_jobs * args.num_tasks_in_job, 8), dtype=np.float)

    for jid, job_dag in enumerate(job_dags):
        for tid, task in enumerate(job_dag.tasks):

            wait_time = (curr_time - task.ready_time) if task.ready_time != -1 else 0
            wait_time /= (curr_time + EPS)

            task_predecessors = np.sum(get_predecessor_repr(task, env_storage)) / 10

            if task.PE_ID == -1:
                task_PE_ID_repr = np.zeros(NUM_RESOURCE_MATRIX, dtype=np.float)
            elif 0 <= task.PE_ID < NUM_RESOURCE_MATRIX:
                task_PE_ID_repr = get_one_hot_repr(task.PE_ID, NUM_RESOURCE_MATRIX)
            else:
                raise Exception()

            statuses = np.zeros((3), dtype=np.float)
            if task in wait_task:
                statuses[0] = 1
            elif task in running:
                statuses[1] = 1
            elif task in outstanding:
                statuses[2] = 1

            processed = np.concatenate((task_PE_ID_repr, [wait_time], [task_predecessors], statuses))

            task_processed[jid*10 + tid, :] = processed

    return np.concatenate(
        (np.asarray(b_level), num_children_tasks, job_wait_time, task_processed.flatten())
    ).reshape(1, -1)


def one_hot_preprocess(gym_env, job_dags):
    # Attribute: Used? Description.

    # ------------------------ USED ----------------------------------------------------------- #
    # ID: unique per task. Useful. Assume log(#tasks)-bit representation.
    # PE_ID: Useful. One hot representation, size: # PEs.
    # deadline: Useful. Especially for the reward. Assume log(#timestamps)-bit representation.
    # est: Useful. Earliest time a task can task. Assume log(#timestamps)-bit representation.
    # start_time, finish_time: Useful. Denotes the timestamp the task ended at.
    #                          Assume #timestamps-bit representation.
    # TOTAL: log(MAX_NUM_TASKS) + #PEs + log(MAX_TIMESTAMP) * 4 + #Statuses = 10 + 4 + 10 * 4 + 4 = 58

    # ----- TO BE ADDED  ------------------------------------------------------------------------ #  TODO: Add these.
    # predecessors: Ignored for now. Useful. Denotes tasks that need to be completed?
    #               Assume n*log(#tasks)-bit representation + LSTM?
    # ready_wait_times: Ignored for now. But useful? Seems to denote time delays due to predecessors.
    #                   Assume n*log(#timestamps)-bit representation + LSTM?

    # ----- OTHER VARIABLES TO CONSIDER, --------------------------------------------- # TODO: Check these.
    # PE_to_PE_wait_time: Ignored for now.
    # base_ID: Ignored for now.
    # execution_wait_times: Ignored for now.
    # head, tail: Ignored for now.
    # jobID, job_start, jobname: Ignored for now.
    # name: Ignored for now.
    # time_elapsed_time_max_frew: Ignored for now.
    # timestamp: Ignored for now. Not sure what this is.
    args = gym_env.args
    env_storage = gym_env.env_storage
    
    state = np.zeros((args.max_num_jobs * args.num_tasks_in_job, get_repr_size()), dtype=np.float)

    for jid, job_dag in enumerate(job_dags):
        for tid, task in enumerate(job_dag.tasks):

            wait_time = (curr_time - task.ready_time) if task.ready_time != -1 else 0
            task_wait_time_repr = get_binary_repr(wait_time, MAX_WAIT_TIME)

            job_inject_time = (curr_time - task.job_start) if task.job_start != -1 else 0
            job_injected_time_repr = get_binary_repr(job_inject_time, MAX_WAIT_TIME)

            task_predecessor_repr = get_predecessor_repr(task, env_storage)

            task_start_time_repr = get_binary_repr(task.start_time, MAX_TIMESTAMP, allow_undefined=True)

            status = generate_task_status(env_storage, task)

            processed = np.concatenate(
                (task_predecessor_repr, task_start_time_repr, task_wait_time_repr, job_injected_time_repr, status)
            )

            state[jid*10 + tid, :] = processed

    return state


def generate_task_status(env_storage, task):
    status = np.zeros((6,), dtype=np.float)
    if task in env_storage.TaskQueues.ready.list:
        status[0] = 1
    elif task in env_storage.TaskQueues.executable.list:
        status[1] = 1
    elif task in env_storage.TaskQueues.wait_ready.list:
        status[2] = 1
    elif task in env_storage.TaskQueues.running.list:
        status[3] = 1
    elif task in env_storage.TaskQueues.completed.list:
        status[4] = 1
    elif task in env_storage.TaskQueues.outstanding.list:
        status[5] = 1

    return status


def generate_task_status_pairs(env_storage):
    all_tasks = env_storage.TaskQueues.ready.list + env_storage.TaskQueues.running.list + \
                env_storage.TaskQueues.completed.list + env_storage.TaskQueues.outstanding.list + \
                env_storage.TaskQueues.executable.list + env_storage.TaskQueues.wait_ready.list

    num_total_tasks = len(all_tasks)

    status = np.zeros((6,))
    status[0] = len(env_storage.TaskQueues.ready.list) / num_total_tasks
    status[1] = len(env_storage.TaskQueues.executable.list) / num_total_tasks
    status[2] = len(env_storage.TaskQueues.wait_ready.list) / num_total_tasks
    status[3] = len(env_storage.TaskQueues.running.list) / num_total_tasks
    status[4] = len(env_storage.TaskQueues.completed.list) / num_total_tasks
    status[5] = len(env_storage.TaskQueues.outstanding.list) / num_total_tasks

    return status
