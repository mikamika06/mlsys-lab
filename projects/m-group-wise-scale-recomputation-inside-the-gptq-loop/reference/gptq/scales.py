import numpy as np


def compute_group_scales(w: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    max_val = float(2 ** (bits - 1) - 1)
    rows, cols = w.shape
    num_groups = (cols + group_size - 1) // group_size
    scales = np.zeros((rows, num_groups), dtype=np.float32)
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, cols)
        sub = w[:, start:end]
        maxs = np.max(np.abs(sub), axis=1)
        scales[:, g] = np.maximum(maxs / max_val, 1e-8)
    return scales
