import numpy as np
from heal.mask import MaskManager


class Pruner:
    """Handles structured/unstructured pruning and baseline metric extraction."""

    def __init__(self, model):
        self.model = model
        self.mask_mgr = MaskManager(model.weights)

    def prune_by_magnitude(self, target_sparsity):
        self.mask_mgr.create_magnitude_mask(target_sparsity)
        self.mask_mgr.apply_mask()
        return self.mask_mgr

    def get_baseline_stats(self, X, y):
        acc, loss = self.model.evaluate(X, y)
        return {
            "acc": float(acc),
            "loss": float(loss),
            "sparsity": float(self.mask_mgr.get_sparsity()),
        }
