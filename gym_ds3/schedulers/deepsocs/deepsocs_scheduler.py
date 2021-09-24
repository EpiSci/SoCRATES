import copy

import networkx as nx
import numpy as np

from gym_ds3.schedulers.heuristic.heft import heft, dag_merge
from gym_ds3.schedulers.deepsocs.msg_path import Postman, get_unfinished_nodes_summ_mat


class Deepsocs(object):

    def __init__(self, args, model, sess):
        self.args = args
        self.model = model
        self.sess = sess
        
        self.node_input_dim = self.args.node_input_dim
        self.job_input_dim = self.args.job_input_dim
        
    def reset(self):
        self.simleng = self.args.simulation_length
        self.postman = Postman(self.args)

    def schedule(self, env, state, batch_size, train):
        """
        Input: state - a tuple of (job_dags, action_map, env_storage, rts, pes, env.now)
        Returns: scheduled_tasks - a dictionary of tasks and pes
        """
        scheduled_tasks = {}

        batch_buf = {
            'obs_change': [],
            'node_inputs': [],
            'gcn_mats': [],
            'gcn_masks': [],
            'summ_mats': [],
            'running_dag_mat': [],
            'dag_summ_backward': [],
            'node_valid_mask': [],
            'wall_time': [],
            'a1': []
            }

        job_dags_list, action_map, env_storage, pes, env_timestep = state
        node_inputs, job_dags_list, num_source_pe, pes, action_map = self.translate_state(state)
        
        summ_mats = get_unfinished_nodes_summ_mat(job_dags_list)
        
        # preprocess job_dags for gcn.
        gcn_mats, gcn_masks, dag_summ_backward_map, running_dags_mat, job_dags_changed = \
            self.postman.get_msg_path(job_dags_list)

        self.assigned_pes = [0 if pes[i].idle else 1 for i in range(len(pes) - 1)]

        check_rts = copy.deepcopy(env_storage.TaskQueues.ready.list)
        
        nn_tasks = []

        while check_rts:
            node_valid_mask = self.get_valid_mask(job_dags_list, check_rts) #env.rts)
            feed_dict = {i: d for i, d in zip(
                [self.model['batch']] + [self.model['o_ph']] + self.model['adj_mats'] +
                self.model['masks'] + self.model['summ_mats'] + [self.model['dag_summ_backward']] +
                [self.model['node_valid_mask']],
                [batch_size] + [node_inputs] + gcn_mats +
                gcn_masks + [summ_mats, running_dags_mat] + [dag_summ_backward_map] +
                [node_valid_mask])}

            task, t_pi = self.sess.run([self.model['a1'], self.model['t_pi']], feed_dict=feed_dict)

            task = [jd.jobID for jd in job_dags_list][task[0] // 10] * 10 + task[0] % 10

            t_pi = np.expand_dims(np.transpose(t_pi), axis=0)

            # Add to buffer
            if train:
                batch_buf['obs_change'].append(job_dags_changed)
                batch_buf['node_inputs'].append(node_inputs)
                batch_buf['summ_mats'].append(summ_mats)
                batch_buf['running_dag_mat'].append(running_dags_mat)
                batch_buf['node_valid_mask'].append(node_valid_mask)
                batch_buf['wall_time'].append(env_timestep)
                batch_buf['a1'].append(t_pi)
                if job_dags_changed:
                    batch_buf['gcn_mats'].append(gcn_mats)
                    batch_buf['gcn_masks'].append(gcn_masks)
                    batch_buf['dag_summ_backward'].append(dag_summ_backward_map)

            task = action_map[task]
            nn_tasks.append(task.ID)

            for _task in check_rts:
                if _task.ID == task.ID:
                    task = _task
                    break

            for rt in check_rts:
                if rt.ID == task.ID:
                    check_rts.remove(rt)

        # Start eft method
        dag, computation_matrix, running_tasks \
            = self.dag_preprocess(
            env_storage, env_storage.TaskQueues.ready.list, env.now(), env.resource_matrix_list)

        dict_output = self.eft_selects_pe(
            list(dag.nodes().keys()), env.now(), dag, computation_matrix, env.comm_band, running_tasks)

        for task in env_storage.TaskQueues.ready.list:
            scheduled_tasks[task.ID] = dict_output[task.ID]

        return scheduled_tasks, batch_buf

    def get_resource_feat(self, job_dags_list, pes, resource_matrix_list, env_timestep):
        resource_mat = np.zeros((self.args.input_size, self.args.node_input_dim))
        node_idx = 0
        for job_dag in job_dags_list:
            for task in job_dag.tasks:
                for idx, resource in enumerate(resource_matrix_list):
                    if task.name in resource.supported_functionalities:
                        perf_index = resource.supported_functionalities.index(task.name)
                        resource_mat[node_idx, idx] = resource.performance[perf_index]
                for idx, pe in enumerate(pes):
                    resource_mat[node_idx, len(resource_matrix_list) + idx] = \
                        pe.task_expected_total_time - env_timestep
                node_idx += 1

        return resource_mat

    def translate_state(self, state):
        """
        Translate the observation to matrix form
        """
        job_dags_list, action_map, env_storage, pes, env_timestep = state

        num_source_pe = sum(1 for pe in pes if not pe.idle)

        # compute total number of nodes
        total_num_tasks = int(np.sum(job_dag.num_nodes for job_dag in job_dags_list))

        # job and node inputs to feed
        node_inputs = np.zeros([total_num_tasks, self.node_input_dim])
        job_inputs = np.zeros([len(job_dags_list), self.job_input_dim])

        # node_inputs: [total_num_tasks, 5]
        # pe_inputs: [num_pe, total_num_tasks]
        # pe_inputs x node_inputs = pe_feature
        #     [num_pe, pe_input_dim] -> [num_pe, total_num_tasks] x [total_num_tasks, node_input_dim]

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

    def get_valid_mask(self, job_dags, rts):
        total_num_nodes = int(np.sum([job_dag.num_nodes for job_dag in job_dags]))

        node_valid_mask = np.zeros([1, total_num_nodes])

        job_dag_ids = [job_dag.jobID for job_dag in job_dags]

        for rt in rts:
            node_valid_mask[0, job_dag_ids.index(rt.ID // 10) * 10 + rt.base_ID] = 1

        return node_valid_mask

    def dag_preprocess(self, env_storage, prep_tasks, timestep, resource_matrix_list):
        computation_dict = {}
        dag = nx.DiGraph()

        for task in prep_tasks:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            for idx, resource in enumerate(resource_matrix_list):
                if task.name in resource.supported_functionalities:
                    perf_index = resource.supported_functionalities.index(task.name)
                    computation_dict[task.ID].append(resource.performance[perf_index])
                else:
                    computation_dict[task.ID].append(np.inf)

        dag = dag_merge.merge_dags(dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT,
                                   skip_relabeling=True)
        computation_dict[max(dag) - 1] = np.zeros((1, len(resource_matrix_list)))
        computation_dict[max(dag)] = np.zeros((1, len(resource_matrix_list)))
        computation_matrix = np.empty((max(dag) + 1, len(resource_matrix_list)))

        running_tasks = {}
        for idx in range(len(resource_matrix_list)):
            running_tasks[idx] = []

        for task in env_storage.TaskQueues.running.list:
            executing_resource = resource_matrix_list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for task in env_storage.TaskQueues.executable.list:
            executing_resource = resource_matrix_list[task.PE_ID]
            task_id = task.ID
            if len(running_tasks[task.PE_ID]) is not 0:
                task_start = running_tasks[task.PE_ID][-1].end
            else:
                task_start = timestep
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for key, val in computation_dict.items():
            computation_matrix[key, :] = val

        return dag, computation_matrix, running_tasks

    def eft_selects_pe(self, nn_tasks, timestep, dag, computation_matrix, communication_matrix, proc_schedules):
        from collections import namedtuple
        from types import SimpleNamespace
        from math import inf
        from gym_ds3.schedulers.heuristic.heft.heft import _compute_eft

        ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

        if proc_schedules == None:
            proc_schedules = {}

        _self = {
            'computation_matrix': computation_matrix,
            'communication_matrix': communication_matrix,
            'task_schedules': {},
            'proc_schedules': proc_schedules,
            'numExistingJobs': 0,
            'time_offset': timestep,
            'root_node': None
        }
        _self = SimpleNamespace(**_self)

        # Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

        for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
            _self.task_schedules[i] = None

        for i in range(len(_self.communication_matrix)):
            if i not in _self.proc_schedules:
                _self.proc_schedules[i] = []

        for proc in proc_schedules:
            for schedule_event in proc_schedules[proc]:
                _self.task_schedules[schedule_event.task] = schedule_event

        # Nodes with no successors cause the any expression to be empty
        root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
        assert len(root_node) == 1, f"Expected a single root node, found {len(root_node)}"
        root_node = root_node[0]
        _self.root_node = root_node

        if nn_tasks[0] != root_node:
            idx = nn_tasks.index(root_node)
            nn_tasks[idx], nn_tasks[0] = nn_tasks[0], nn_tasks[idx]

        for node in nn_tasks:
            if _self.task_schedules[node] is not None:
                continue

            minTaskSchedule = ScheduleEvent(node, inf, inf, -1)

            for proc in range(len(communication_matrix)):
                taskschedule = _compute_eft(_self, dag, node, proc)

                if taskschedule.end < minTaskSchedule.end:
                    minTaskSchedule = taskschedule

            _self.task_schedules[node] = minTaskSchedule
            _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
            _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc],
                                                                key=lambda schedule_event: schedule_event.end)

            for proc in range(len(_self.proc_schedules)):
                for job in range(len(_self.proc_schedules[proc]) - 1):
                    first_job = _self.proc_schedules[proc][job]
                    second_job = _self.proc_schedules[proc][job + 1]

                    assert first_job.end <= second_job.start, \
                        f"Jobs on a particular processor must finish before the next can begin, but job " \
                            f"{first_job.task} on processor {first_job.proc} ends at {first_job.end} and " \
                            f"its successor {second_job.task} starts at {second_job.start}"

        dict_output = {}
        for proc_num, proc_tasks in _self.proc_schedules.items():
            for idx, task in enumerate(proc_tasks):
                if idx > 0 and (proc_tasks[idx - 1].end - proc_tasks[idx - 1].start > 0):
                    dict_output[task.task] = (proc_num, idx, [proc_tasks[idx - 1].task])
                else:
                    dict_output[task.task] = (proc_num, idx, [])

        return dict_output
