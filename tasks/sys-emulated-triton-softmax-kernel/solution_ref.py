import math
import numpy as np


def softmax_kernel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    out = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        max_val = X[i, 0]
        for j in range(1, d):
            if X[i, j] > max_val:
                max_val = X[i, j]
        row_sum = 0.0
        for j in range(d):
            val = math.exp(X[i, j] - max_val)
            out[i, j] = val
            row_sum += val
        for j in range(d):
            out[i, j] /= row_sum
    return out
