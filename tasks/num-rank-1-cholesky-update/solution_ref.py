import numpy as np


def rank1_cholesky_update(L, x):
    L = np.array(L, dtype=np.float64, copy=True)
    x = np.array(x, dtype=np.float64, copy=True)
    n = L.shape[0]

    for k in range(n):
        r = np.sqrt(L[k, k] * L[k, k] + x[k] * x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            old_col = L[k + 1:, k].copy()
            L[k + 1:, k] = (old_col + s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]

    return L
