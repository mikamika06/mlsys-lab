import math
import numpy as np

def reconstruct_inverse_hessian(A: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Compute the inverse of H = A @ A.T + λ I using Cholesky factorisation.
    Returns an (n,n) array of dtype float64.
    """
    n = A.shape[0]
    d = A.shape[1]

    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(d):
                val += float(A[i, k]) * float(A[j, k])
            if i == j:
                val += lambda_reg
            H[i, j] = val

    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = math.sqrt(H[i, i] - s)
            else:
                L[i, j] = (H[i, j] - s) / L[j, j]

    X = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(j, n):
            s = 0.0
            for k in range(j, i):
                s += L[i, k] * X[k, j]
            rhs = 1.0 if i == j else 0.0
            X[i, j] = (rhs - s) / L[i, i]

    inv_H = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(n - 1, -1, -1):
            s = 0.0
            for k in range(i + 1, n):
                s += L[k, i] * inv_H[k, j]
            inv_H[i, j] = (X[i, j] - s) / L[i, i]

    return inv_H
