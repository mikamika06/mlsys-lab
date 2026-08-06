class RadixEvictionManager:
    def __init__(self, capacity):
        raise NotImplementedError

    def evict(self, tree_root, current_usage):
        raise NotImplementedError
