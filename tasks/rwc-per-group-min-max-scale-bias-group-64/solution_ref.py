import numpy as np

def compute_group_params(weights: np.ndarray,
                         group_size: int = 64,
                         bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    n_groups = len(w) // group_size
    reshaped = w.reshape(n_groups, group_size)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    scale = (maxs - mins) / (2**bits - 1)
    bias = mins
    return scale.astype(np.float64), bias.astype(np.float64)
