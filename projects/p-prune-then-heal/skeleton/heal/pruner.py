import numpy as np


class Pruner:
    """Handles structured/unstructured pruning and baseline metric extraction."""

    def __init__(self, model):
        raise NotImplementedError

    def prune_by_magnitude(self, target_sparsity):
        raise NotImplementedError

    def get_baseline_stats(self, X, y):
        raise NotImplementedError
