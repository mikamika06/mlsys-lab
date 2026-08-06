from __future__ import annotations

import math
import numpy as np

SYM_TOL = 1e-10


def cholesky_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> np.ndarray | None:
    """Hand-rolled Cholesky; returns lower-triangular L with L @ L.T == A, or None."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    n = A.shape[0]
    max_diff = 0.0
    for i in range(n):
        for j in range(n):
            diff = abs(float(A[i, j]) - float(A[j, i]))
            if diff > max_diff:
                max_diff = diff

    if max_diff > sym_tol:
        return None

    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            dot = 0.0
            for k in range(j):
                dot += float(L[i, k]) * float(L[j, k])
            s = float(A[i, j]) - dot
            if i == j:
                if s <= 0.0:
                    return None
                L[i, j] = math.sqrt(s)
            else:
                L[i, j] = s / float(L[j, j])
    return L


def is_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> bool:
    """True iff A is symmetric within sym_tol and positive definite."""
    return cholesky_spd(A, sym_tol) is not None
