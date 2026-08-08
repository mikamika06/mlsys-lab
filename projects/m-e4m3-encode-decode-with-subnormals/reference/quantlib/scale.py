import numpy as np


def compute_scale(x):
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x))
    if m == 0:
        return 1.0
    return float(m / 448.0)
