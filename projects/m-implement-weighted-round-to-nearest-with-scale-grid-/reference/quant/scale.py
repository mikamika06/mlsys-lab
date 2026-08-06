import numpy as np


def optimal_scale(weights, imatrix, q_min, q_max):
    weights = np.asarray(weights, dtype=np.float32)
    imatrix = np.asarray(imatrix, dtype=np.float32)
    max_val = max(abs(q_min), abs(q_max))
    max_w = np.max(np.abs(weights))
    if max_w == 0:
        return 1.0
    initial_scale = max_w / max_val
    q = np.clip(np.round(weights / initial_scale), q_min, q_max)
    num = np.sum(imatrix * weights * q)
    den = np.sum(imatrix * q * q)
    if den == 0:
        return float(initial_scale)
    return float(num / den)
