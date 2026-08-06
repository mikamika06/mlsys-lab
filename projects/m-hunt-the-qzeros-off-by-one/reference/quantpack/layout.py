import numpy as np

def apply_permutation(weight: np.ndarray, g_idx: np.ndarray):
    perm = np.argsort(g_idx)
    return weight[perm, :], perm

def invert_permutation(sorted_weight: np.ndarray, perm: np.ndarray):
    inv_perm = np.argsort(perm)
    return sorted_weight[inv_perm, :]
