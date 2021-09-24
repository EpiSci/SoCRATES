import numpy as np
import networkx as nx

from gym_ds3.schedulers.heuristic.heft import heft, dag_merge


class HEFT_RT(object):
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
        env_storage = self.env.env_storage
        list_of_ready = env_storage.TaskQueues.ready.list

        if not list_of_ready:
            return

        computation_dict = {}
        dag = nx.DiGraph()
        for task in list_of_ready:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            for idx, resource in enumerate(self.env.resource_matrix_list):
                if task.name in resource.supported_functionalities:
                    perf_index = resource.supported_functionalities.index(task.name)
                    computation_dict[task.ID].append(resource.performance[perf_index])
                else:
                    computation_dict[task.ID].append(np.inf)
        dag = dag_merge.merge_dags(
            dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT, skip_relabeling=True)

        computation_dict[max(dag) - 1] = np.zeros((1, len(self.env.resource_matrix_list)))
        computation_dict[max(dag)] = np.zeros((1, len(self.env.resource_matrix_list)))
        computation_matrix = np.empty((max(dag) + 1, len(self.env.resource_matrix_list)))

        running_tasks = {}
        for idx in range(len(self.env.resource_matrix_list)):
            running_tasks[idx] = []

        for task in env_storage.TaskQueues.running.list:
            executing_resource = self.env.resource_matrix_list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for task in env_storage.TaskQueues.executable.list:
            executing_resource = self.env.resource_matrix_list[task.PE_ID]
            task_id = task.ID
            if len(running_tasks[task.PE_ID]) is not 0:
                task_start = running_tasks[task.PE_ID][-1].end
            else:
                task_start = self.env.now()
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for key, val in computation_dict.items():
            computation_matrix[key, :] = val

        _, _, dict_output, _ = heft.schedule_dag(
            dag,
            computation_matrix=computation_matrix,
            communication_matrix=self.env.resource_matrix.comm_band,
            time_offset=self.env.now(),
            proc_schedules=running_tasks,
            relabel_nodes=False
        )
        
        return dict_output
    
    def select_action(self, env_storage, mapped_tasks):
        for task in env_storage.TaskQueues.ready.list:
            if task.ID in list(mapped_tasks.keys()):
                task.PE_ID = mapped_tasks[task.ID][0]
                task.dynamic_dependencies = mapped_tasks[task.ID][2]

    def sort_tasks_by_ranku(self, env_storage):
        list_of_ready = env_storage.TaskQueues.ready.list

        computation_dict = {}
        dag = nx.DiGraph()
        for task in list_of_ready:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            for idx, resource in enumerate(self.env.resource_matrix_list):
                if task.name in resource.supported_functionalities:
                    perf_index = resource.supported_functionalities.index(task.name)
                    computation_dict[task.ID].append(resource.performance[perf_index])
                else:
                    computation_dict[task.ID].append(np.inf)
        dag = dag_merge.merge_dags(dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT,
                                   skip_relabeling=True)

        computation_dict[max(dag) - 1] = np.zeros((1, len(self.env.resource_matrix_list)))
        computation_dict[max(dag)] = np.zeros((1, len(self.env.resource_matrix_list)))
        computation_matrix = np.empty((max(dag) + 1, len(self.env.resource_matrix_list)))

        running_tasks = {}
        for idx in range(len(self.env.resource_matrix_list)):
            running_tasks[idx] = []

        for task in env_storage.TaskQueues.running.list:
            executing_resource = self.env.resource_matrix_list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for task in env_storage.TaskQueues.executable.list:
            executing_resource = self.env.resource_matrix_list[task.PE_ID]
            task_id = task.ID
            if len(running_tasks[task.PE_ID]) is not 0:
                task_start = running_tasks[task.PE_ID][-1].end
            else:
                task_start = self.env.now()
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        for key, val in computation_dict.items():
            computation_matrix[key, :] = val

        ranku_priorities, ranks = heft.compute_ranku_priorities(
            dag,
            computation_matrix=computation_matrix,
            communication_matrix=self.env.resource_matrix.comm_band,
            time_offset=self.env.now(),
            proc_schedules=running_tasks,
            relabel_nodes=False
        )

        sorted_tasks = []
        rid_rt_pairs = {rt.ID: rt for rt in env_storage.TaskQueues.ready.list}
        for rank in ranku_priorities:
            if rank not in rid_rt_pairs.keys():
                continue
            else:
                sorted_tasks.append(rid_rt_pairs[rank])

        return sorted_tasks
