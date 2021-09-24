import networkx as nx
import numpy as np

import torch

from gym_ds3.envs.core.node import Node
from gym_ds3.envs.utils.helper_dict import OrderedSet


class JobDAG(object):
    def __init__(self, job):
        self.job = job
        self.jobID = self.job.task_list[0].jobID
        self.commvol = self.job.comm_vol

        self.tasks = self.job.task_list

        self.adj_mat = self.get_adj_mat(self.tasks)

        self.arrived = False  # dag is arrived
        self.is_completed = False
        self.is_running = False

        self.start_exec_time = np.inf  # dag start time
        self.start_inject_time = np.inf  # dag inject time
        self.completion_time = np.inf  # dag finish time

        # Dependency graph (num_tasks, num_tasks)
        self.predecessor = self.predecessors(self.tasks)

        # The features of Node : jobID, taskID, status, deadline, start time, finish time, est
        self.nodes = self.get_nodes(self.tasks, self.adj_mat)
        self.num_nodes = len(self.job.task_list)

        self.frontier_nodes = OrderedSet()
        for node in self.nodes:
            if node.is_schedulable():
                self.frontier_nodes.add(node)

    def get_adj_mat(self, tasks):
        adj = nx.DiGraph(self.commvol)
        adj.remove_edges_from(
            # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist"
            [edge for edge in adj.edges() if adj.get_edge_data(*edge)['weight'] == '0.0']
        )
        nx.relabel_nodes(adj, lambda idx: idx, copy=False)
        adj = from_networkx(adj)
        mat = np.zeros((len(tasks), len(tasks)))
        index_list = adj['edge_index'].transpose(0, 1)  # .T does not work with pytorch > v1.1
        for i in range(len(index_list)):
            mat[index_list[i][0]][index_list[i][1]] = 1
        return mat

    def get_nodes(self, tasks, adj_mat):
        nodes = [Node(task) for task in tasks]

        for i in range(len(tasks)):
            for j in range(len(tasks)):
                if adj_mat[i, j] == 1:
                    nodes[i].child_nodes.append(nodes[j])
                    nodes[j].parent_nodes.append(nodes[i])

        return nodes

    def predecessors(self, tasks):
        dependency = np.zeros((len(tasks), len(tasks)))
        for idx, node in enumerate(tasks):
            for predecessorNode in node.predecessors:
                dependency[idx][predecessorNode % len(tasks)] = 1.
        return dependency


# Modified from https://github.com/rusty1s/pytorch_geometric/blob/e6b8d6427ad930c6117298006d7eebea0a37ceac/torch_geometric/utils/convert.py#L108
def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data['num_nodes'] = G.number_of_nodes()

    return data
