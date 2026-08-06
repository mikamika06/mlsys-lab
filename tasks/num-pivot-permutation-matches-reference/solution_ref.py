import math
import numpy as np


def lu_pivot_indices(A: np.ndarray) -> np.ndarray:
    """LAPACK-style partial-pivoting swap vector: piv[k] is the row A's row k
    was exchanged with at elimination step k."""
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    piv = np.zeros(n, dtype=np.int64)
    for k in range(n):
        if k < n - 1:
            max_val = -1.0
            p = k
            for i in range(k, n):
                val = math.fabs(A[i, k])
                if val > max_val:
                    max_val = val
                    p = i
        else:
            p = k
        piv[k] = p
        if p != k:
            for j in range(n):
                tmp = A[k, j]
                A[k, j] = A[p, j]
                A[p, j] = tmp
        if k < n - 1 and A[k, k] != 0.0:
            akk = A[k, k]
            for i in range(k + 1, n):
                factor = A[i, k] / akk
                A[i, k] = factor
                for j in range(k + 1, n):
                    A[i, j] -= factor * A[k, j]
    return piv
