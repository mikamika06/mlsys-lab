import numpy as np


def stable_log_add_exp(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.maximum(a, b)
    return m + np.log1p(np.exp(-np.abs(a - b)))
