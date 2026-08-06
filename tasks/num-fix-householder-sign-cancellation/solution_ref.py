import math
import numpy as np


def householder_fixed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    sum_sq = 0.0
    for i in range(m):
        val = x[i]
        sum_sq += val * val
    norm = math.sqrt(sum_sq)
    if norm == 0:
        return np.eye(m, dtype=np.float64)
    sign = -1.0 if x[0] >= 0 else 1.0
    alpha = sign * norm
    v = [x[i] for i in range(m)]
    v[0] -= alpha
    beta = 0.0
    for i in range(m):
        beta += v[i] * v[i]
    if beta == 0:
        return np.eye(m, dtype=np.float64)
    H = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            val = 1.0 if i == j else 0.0
            val -= 2.0 * v[i] * v[j] / beta
            H[i, j] = val
    return H
