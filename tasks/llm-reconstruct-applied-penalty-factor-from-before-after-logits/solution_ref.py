import numpy as np


def reconstruct_penalty_factor(before, after, affected_indices):
    x = np.asarray(before, dtype=np.float64)[np.asarray(affected_indices)]
    y = np.asarray(after, dtype=np.float64)[np.asarray(affected_indices)]
    return float(np.exp(np.mean(np.log(x / y))))
