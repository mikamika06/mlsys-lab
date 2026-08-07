import numpy as np


def compute_max_scale(tensor, qmax=127.0):
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return 1.0
    return float(abs_max / qmax)
