from collections import OrderedDict


def dict2set(source, target):
    for s in source.items():
        target.add(s[1])

    return target


def dict2OrderedSet(source):
    target = OrderedSet()
    for s in source.items():
        target.add(s[1])

    return target


class OrderedSet(object):
    def __init__(self, contents=()):
        self.set = OrderedDict((c, None) for c in contents)

    def __contains__(self, item):
        return item in self.set

    def __iter__(self):
        return iter(self.set.keys())

    def __len__(self):
        return len(self.set)

    def add(self, item):
        self.set[item] = None

    def clear(self):
        self.set.clear()

    def index(self, item):
        idx = 0
        for i in self.set.keys():
            if item == i:
                break
            idx += 1
        return idx

    def pop(self):
        item = next(iter(self.set))
        del self.set[item]
        return item

    def remove(self, item):
        del self.set[item]

    def to_list(self):
        return [k for k in self.set]

    def update(self, contents):
        for c in contents:
            self.add(c)
            