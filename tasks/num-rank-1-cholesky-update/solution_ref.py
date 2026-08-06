import math
import numpy as np


def rank1_cholesky_update(L, x):
    L = np.array(L, dtype=np.float64, copy=True)
    x = np.array(x, dtype=np.float64, copy=True)
    n = L.shape[0]

    for k in range(n):
        r = math.sqrt(L[k, k] * L[k, k] + x[k] * x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            old_col = [L[i, k] for i in range(k + 1, n)]
            x_vals = [x[i] for i in range(k + 1, n)]
            for idx, i in enumerate(range(k + 1, n)):
                new_L = (old_col[idx] + s * x_vals[idx]) / c
                L[i, k] = new_L
                x[i] = c * x_vals[idx] - s * new_L

    return L
