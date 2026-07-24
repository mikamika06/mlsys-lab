"""Reference solution for `num-fix-cholesky-missing-diagonal-sqrt`."""
from __future__ import annotations

import numpy as np


def cholesky(A: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor L such that L @ L.T == A.

    A must be symmetric positive-definite.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L
