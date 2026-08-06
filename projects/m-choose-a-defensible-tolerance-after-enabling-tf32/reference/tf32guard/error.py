import numpy as np


def compute_relative_error(a, b):
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = np.abs(arr_a - arr_b)
    denom = np.maximum(np.abs(arr_b), 1e-12)
    return float(np.max(diff / denom))
