import math
import numpy as np


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_rotate(X, W):
    n = X.shape[1]
    H = _hadamard(n)
    
    sqrt_n = math.sqrt(n)
    Q = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            Q[i, j] = H[i, j] / sqrt_n

    X_float = X.astype(np.float64)
    W_float = W.astype(np.float64)

    m1, k1 = X_float.shape
    n1 = Q.shape[1]
    X_rot = np.empty((m1, n1), dtype=np.float64)
    for i in range(m1):
        for j in range(n1):
            val = 0.0
            for k in range(k1):
                val += X_float[i, k] * Q[k, j]
            X_rot[i, j] = val

    m2, k2 = Q.shape
    n2 = W_float.shape[1]
    Q_T_W = np.empty((m2, n2), dtype=np.float64)
    for i in range(m2):
        for j in range(n2):
            val = 0.0
            for k in range(k2):
                val += Q[k, i] * W_float[k, j]
            Q_T_W[i, j] = val

    return (X_rot, Q_T_W)
