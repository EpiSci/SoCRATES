import copy
import numpy as np

from gym_ds3.envs.misc.action_map import compute_act_map


def preprocess_deepsocs(gym_env):
    job_dags = gym_env.env_storage.job_dags
    job_dags = [jd for jd in job_dags if not jd.is_completed]

    action_map = compute_act_map(gym_env.env_storage.job_dags)

    if gym_env.updated and gym_env.env_storage.TaskQueues.ready.list:
        gym_env.updated = False
    
    obs = (job_dags, action_map, gym_env.env_storage, gym_env.pes, gym_env.now())
    
    return obs


def get_resource_feat(args, x):
    job_dags_list = x['job_dags']
    resource_matrix_list = x['resource_mats']
    pes = x['pes']
    env_timestep = x['timestep']
    EPS = 1e-5
    
    resource_mat = np.zeros((args.input_size, args.node_input_dim))
    resource_mat[:] = 1.
    node_idx = 0
    for job_dag in job_dags_list:
        for task in job_dag.tasks:
            for idx, resource in enumerate(resource_matrix_list):
                if task.name in resource.supported_functionalities:
                    perf_index = resource.supported_functionalities.index(task.name)
                    resource_mat[node_idx, idx] = resource.performance[perf_index]
            for idx, pe in enumerate(pes):
                if not pe.task is None:
                    task_start_time = pe.task.start_time
                else:
                    task_start_time = 0.
                resource_mat[node_idx, len(resource_matrix_list) + idx] = \
                    pe.task_expected_total_time - (env_timestep - task_start_time)
            node_idx += 1

    # normalize values per labels (computation time & remained time)
    for i in range(resource_mat.shape[0]):
        resource_mat[i, 0:3] /= sum(resource_mat[i, 0:3])
        resource_mat[i, 4:] /= (sum(resource_mat[i, 4:]) + EPS)

    return resource_mat


def translate_state(args, state):
    """
    Translate the observation to matrix form
    """
    job_dags_list, action_map, env_storage, rts, pes, env_timestep = state

    num_source_pe = sum(1 for pe in pes if not pe.idle)

    # compute total number of nodes
    total_num_tasks = int(np.sum(job_dag.num_nodes for job_dag in job_dags_list))

    # job and node inputs to feed
    node_inputs = np.zeros([total_num_tasks, args.node_input_dim])
    job_inputs = np.zeros([len(job_dags_list), args.job_input_dim])

    idle_pe_list = [i for i in range(len(pes) - 1) if pes[i].idle]

    # gather job level inputs
    job_idx = 0
    for job_dag in job_dags_list:
        # number of pes in the job
        job_inputs[job_idx, 0] = len(idle_pe_list) / 6.5

        # the current pe belongs to this job or not
        if not job_dag.is_running:
            job_inputs[job_idx, 1] = 2
        else:
            job_inputs[job_idx, 1] = -2

        # number of source pes
        job_inputs[job_idx, 2] = num_source_pe / 6.5  # 1 - job_inputs[job_idx, 0]
        job_idx += 1

    # gather node level inputs
    node_idx = 0
    job_idx = 0
    for job_dag in job_dags_list:

        for task in job_dag.tasks:
            # copy the feature from job_input first
            node_inputs[node_idx, :3] = job_inputs[job_idx, :3]

            if task.PE_ID == -1:
                node_inputs[node_idx, 3:4] = 0.
            else:
                # work on the node
                # 1) Fix -- when task.PE_ID == -1, default pe_commit should be chosen.
                #  Should we do only running tasks for the node_inputs?
                # 2) (total task time - current running task time) * task duration / 100000.
                #  ^Make sure the output is 0.0xxxx
                node_inputs[node_idx, 3] = \
                    pes[task.PE_ID].task_runtime \
                    * pes[task.PE_ID].task_expected_total_time / 10000.

                # number of tasks left
                # (total task time - current running task time) / 200.
                #  ^Make sure the output is 0.0xxxx
                node_inputs[node_idx, 4] = pes[task.PE_ID].task_runtime / 200.

            node_idx += 1

        job_idx += 1

    return node_inputs, job_dags_list, num_source_pe, pes, action_map
