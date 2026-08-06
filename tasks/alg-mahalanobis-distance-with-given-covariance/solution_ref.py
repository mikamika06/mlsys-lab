import math
import numpy as np

def pairwise_mahalanobis(X: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    n, d = X.shape

    Y = np.zeros((n, d), dtype=X.dtype)
    for i in range(n):
        for k in range(d):
            s = 0.0
            for j in range(d):
                s += X[i, j] * cov_inv[j, k]
            Y[i, k] = s

    diag = np.zeros(n, dtype=X.dtype)
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += X[i, k] * Y[i, k]
        diag[i] = s

    out = np.zeros((n, n), dtype=X.dtype)
    for i in range(n):
        for j in range(n):
            xy = 0.0
            for k in range(d):
                xy += X[i, k] * Y[j, k]
            d2 = diag[i] + diag[j] - 2.0 * xy
            if d2 < 0.0:
                d2 = 0.0
            out[i, j] = math.sqrt(d2)

    return out
