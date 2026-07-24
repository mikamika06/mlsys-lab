import numpy as np

def select_top2_mask(weights: np.ndarray) -> np.ndarray:
    groups = weights.reshape(-1, 4)
    abs_groups = np.abs(groups)
    idxs = np.argsort(-abs_groups, axis=1)[:, :2]
    row_indices = np.repeat(np.arange(groups.shape[0]), 2)
    col_indices = idxs.ravel()
    linear_indices = row_indices * 4 + col_indices
    mask = np.zeros_like(weights, dtype=bool)
    mask[linear_indices] = True
    return mask
