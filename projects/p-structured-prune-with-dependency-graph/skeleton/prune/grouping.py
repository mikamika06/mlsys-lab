class Group:
    """Group of coupled tensor dimensions pruned together."""

    def __init__(self, group_id):
        raise NotImplementedError

    def add_item(self, node, axis):
        raise NotImplementedError


class GroupFinder:
    """Disjoint-set finder for identifying coupled pruning groups."""

    def __init__(self, dep_graph):
        raise NotImplementedError

    def get_pruning_groups(self):
        raise NotImplementedError
