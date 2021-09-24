class Node(object):
    def __init__(self, task):
        self.base_ID = task.base_ID
        self.ID = task.ID
        self.name = task.name
        self.finish_time = task.finish_time
        self.PE_ID = task.PE_ID
        self.child_nodes = []
        self.parent_nodes = []

    def is_schedulable(self):
        if not self.PE_ID == -1:
            return False
        if not self.finish_time == -1:
            return False
        return True
