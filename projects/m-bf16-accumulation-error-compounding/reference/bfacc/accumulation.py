import numpy as np


class Accumulator:
    """Accumulates updates while tracking precision loss."""

    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.fp32_master = np.zeros(shape, dtype=np.float32)
        self.bf16_naive = np.zeros(shape, dtype=np.float32)

    def update(self, delta):
        delta_fp32 = np.array(delta, dtype=np.float32)
        delta_bf16 = delta_fp32.astype(np.float16).astype(np.float32)
        self.fp32_master += delta_fp32
        self.bf16_naive = (self.bf16_naive.astype(np.float16) + delta_bf16.astype(np.float16)).astype(np.float32)

    def get_values(self):
        return self.bf16_naive, self.fp32_master


def compute_relative_error(naive_arr, compensated_arr):
    n_arr = np.array(naive_arr, dtype=np.float64)
    c_arr = np.array(compensated_arr, dtype=np.float64)
    norm_diff = np.linalg.norm(n_arr - c_arr)
    norm_comp = np.linalg.norm(c_arr)
    if norm_comp == 0.0:
        return 0.0
    return float(norm_diff / norm_comp)
