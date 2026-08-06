import numpy as np

def optimize_scale(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return 1.0
    return float(max_val / 448.0)
