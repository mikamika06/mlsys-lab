import numpy as np


class MaskManager:
    """Manages magnitude pruning masks and gradient zeroing for weights."""

    def __init__(self, weights):
        raise NotImplementedError

    def create_magnitude_mask(self, sparsity_target):
        raise NotImplementedError

    def apply_mask(self):
        raise NotImplementedError

    def mask_gradients(self, grads):
        raise NotImplementedError

    def get_sparsity(self):
        raise NotImplementedError
