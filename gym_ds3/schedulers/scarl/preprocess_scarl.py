import numpy as np

from gym_ds3.schedulers.socrates.preprocess_socrates import get_num_dependency, get_one_hot_repr, get_predecessor_repr, NUM_RESOURCE_MATRIX


def preprocess_scarl(gym_env, task_id):
    job_dags = [jd for jd in gym_env.env_storage.job_dags if not jd.is_completed]

    if gym_env.updated and gym_env.env_storage.TaskQueues.ready.list:
        gym_env.updated = False

    _, num_remained_dependency = get_num_dependency(gym_env.env_storage)
    task_obs = floating_preprocess_scarl(gym_env, job_dags, num_remained_dependency, task_id)
    
    pe_obs = get_pe_features(gym_env.pes)

    return task_obs, pe_obs


def floating_preprocess_scarl(gym_env, job_dags, num_remained_dependency, task_id=0):
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
        if jd.job.name is 'Top_1':
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

    task_processed = np.zeros((1, 8), dtype=np.float)

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
            if task_id == task.ID: # Only uses current task feature
                task_processed[:, :] = processed
                
    task_obs = np.concatenate((np.asarray(b_level), num_children_tasks, job_wait_time, task_processed.flatten())).reshape(1, -1)

    return task_obs


def get_pe_features(pes):
    pe_features = np.zeros((3, 10), dtype=np.float)
    for i, pe in enumerate(pes):
        if i == 3: # Memory
            break
        if pe.type == 'CPU':
            pe_features[i, :3] = np.array([1, 0, 0])
        elif pe.type == 'ACC':
            pe_features[i, :3] = np.array([0, 1, 0])
        elif pe.type == 'MEM':
            pe_features[i, :3] = np.array([0, 0, 1])
        pe_features[i, 3] = pe.capacity
        pe_features[i, 4] = pe.std
        pe_features[i, 5] = pe.available_time
        if pe.idle:
            pe_features[i, 6] = 1.
        else:
            pe_features[i, 6] = 0.
        pe_features[i, 7] = pe.busy_dur
        pe_features[i, 8] = pe.task_runtime
        pe_features[i, 9] = pe.task_expected_total_time

    return pe_features
