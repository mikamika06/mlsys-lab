import numpy as np


def lu_pivot_indices(A: np.ndarray) -> np.ndarray:
    """LAPACK-style partial-pivoting swap vector: piv[k] is the row A's row k
    was exchanged with at elimination step k."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    piv = np.zeros(n, dtype=np.int64)
    for k in range(n):
        if k < n - 1:
            p = k + int(np.argmax(np.abs(A[k:, k])))
        else:
            p = k
        piv[k] = p
        if p != k:
            A[[k, p], :] = A[[p, k], :]
        if k < n - 1 and A[k, k] != 0.0:
            factors = A[k + 1:, k] / A[k, k]
            A[k + 1:, k] = factors
            A[k + 1:, k + 1:] -= np.outer(factors, A[k, k + 1:])
    return piv
