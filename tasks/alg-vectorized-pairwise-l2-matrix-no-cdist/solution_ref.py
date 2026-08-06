import numpy as np


def pairwise_l2_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    n, d = X.shape
    m = Y.shape[0]

    X_norm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(d):
            s += X[i, k] ** 2
        X_norm[i] = s

    Y_norm = np.zeros(m, dtype=np.float64)
    for j in range(m):
        s = 0.0
        for k in range(d):
            s += Y[j, k] ** 2
        Y_norm[j] = s

    result = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k in range(d):
                dot += X[i, k] * Y[j, k]
            result[i, j] = X_norm[i] + Y_norm[j] - 2.0 * dot

    return result
