import numpy as np


class Accumulator:
    """Accumulates updates while tracking precision loss."""

    def __init__(self, shape, dtype=np.float32):
        raise NotImplementedError

    def update(self, delta):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError


def compute_relative_error(naive_arr, compensated_arr):
    raise NotImplementedError
