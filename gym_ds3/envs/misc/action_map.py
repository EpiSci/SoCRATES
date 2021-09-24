class two_way_unordered_map(object):
    def __init__(self):
        self.map = {}
        self.inverse_map = {}

    def __setitem__(self, key, value):
        self.map[key] = value
        self.inverse_map[value] = key
        # keys and values should be unique
        assert len(self.map) == len(self.inverse_map)

    def __getitem__(self, key):
        return self.map[key]

    def __len__(self):
        return len(self.map)


def compute_act_map(job_dags):
    # translate action ~ [0, num_nodes_in_all_dags) to node object
    action_map = two_way_unordered_map()
    action = 0
    for job_dag in job_dags:
        for node in job_dag.nodes:
            action_map[action] = node
            action += 1
    return action_map
