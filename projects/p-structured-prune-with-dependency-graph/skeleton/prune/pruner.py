class Pruner:
    def __init__(self, model, groups):
        raise NotImplementedError

    def prune_group(self, group, indices):
        raise NotImplementedError
