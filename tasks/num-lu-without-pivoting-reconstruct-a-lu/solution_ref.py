import numpy as np


def lu_no_pivot(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        for j in range(k, n):
            dot_product = 0.0
            for m in range(k):
                dot_product += L[k, m] * U[m, j]
            U[k, j] = A[k, j] - dot_product

        for i in range(k + 1, n):
            dot_product = 0.0
            for m in range(k):
                dot_product += L[i, m] * U[m, k]
            L[i, k] = (A[i, k] - dot_product) / U[k, k]

    return L, U
