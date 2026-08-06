import numpy as np
import math

def hessian_and_inverse(X: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    d = X.shape[1]

    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(d):
                s += X[i, k] * X[j, k]
            val = 2.0 * s
            if i == j:
                val += lam
            H[i, j] = val

    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = H[i, i] - s
                L[i, j] = math.sqrt(val)
            else:
                L[i, j] = (H[i, j] - s) / L[j, j]

    I = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        I[i, i] = 1.0

    inv_L = np.zeros((n, n), dtype=np.float64)
    for col in range(n):
        for i in range(n):
            s = 0.0
            for k in range(i):
                s += L[i, k] * inv_L[k, col]
            inv_L[i, col] = (I[i, col] - s) / L[i, i]

    H_inv = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += inv_L[k, i] * inv_L[k, j]
            H_inv[i, j] = s

    return H, H_inv
