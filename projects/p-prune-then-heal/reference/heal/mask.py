import numpy as np


class MaskManager:
    """Manages magnitude pruning masks and gradient zeroing for weights."""

    def __init__(self, weights):
        self.weights = weights
        self.masks = [np.ones_like(w, dtype=bool) for w in weights]

    def create_magnitude_mask(self, sparsity_target):
        for i, w in enumerate(self.weights):
            k = int(w.size * sparsity_target)
            if k > 0:
                threshold = np.partition(np.abs(w).ravel(), k)[k]
                self.masks[i] = np.abs(w) >= threshold
            else:
                self.masks[i] = np.ones_like(w, dtype=bool)

    def apply_mask(self):
        for w, m in zip(self.weights, self.masks):
            w[~m] = 0.0

    def mask_gradients(self, grads):
        for g, m in zip(grads, self.masks):
            g[~m] = 0.0

    def get_sparsity(self):
        total_elems = sum(w.size for w in self.weights)
        total_zeros = sum(int(np.sum(w == 0.0)) for w in self.weights)
        return float(total_zeros) / float(total_elems) if total_elems > 0 else 0.0
