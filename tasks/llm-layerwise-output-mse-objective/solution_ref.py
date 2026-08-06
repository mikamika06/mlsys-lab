import numpy as np


def layerwise_output_mse(W, W_q, X):
    W = np.asarray(W, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    m, k = W.shape
    _, n = X.shape
    total_sq_err = 0.0
    for i in range(m):
        for j in range(n):
            y_ij = 0.0
            y_q_ij = 0.0
            for p in range(k):
                y_ij += W[i, p] * X[p, j]
                y_q_ij += W_q[i, p] * X[p, j]
            diff = y_ij - y_q_ij
            total_sq_err += diff * diff
    return float(total_sq_err / (m * n))
