import numpy as np


def compute_obq_update(w_col, invH_col_idx, invH_val):
    return w_col - (w_col / invH_val) * invH_col_idx
