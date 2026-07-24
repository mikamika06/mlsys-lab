import numpy as np


def lu_no_pivot(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])

        for i in range(k + 1, n):
            L[i, k] = (
                A[i, k] - np.dot(L[i, :k], U[:k, k])
            ) / U[k, k]

    return L, U
