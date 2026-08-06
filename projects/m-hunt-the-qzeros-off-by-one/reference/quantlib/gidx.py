import numpy as np


def apply_gidx(weight, g_idx):
    order = np.argsort(g_idx)
    return weight[:, order]


def invert_gidx(permuted_weight, g_idx):
    order = np.argsort(g_idx)
    inv_order = np.argsort(order)
    return permuted_weight[:, inv_order]
