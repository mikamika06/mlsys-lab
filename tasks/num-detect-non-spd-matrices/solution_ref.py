from __future__ import annotations

import numpy as np

SYM_TOL = 1e-10


def cholesky_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> np.ndarray | None:
    """Hand-rolled Cholesky; returns lower-triangular L with L @ L.T == A, or None."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    if float(np.max(np.abs(A - A.T))) > sym_tol:
        return None

    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = float(A[i, j] - L[i, :j] @ L[j, :j])
            if i == j:
                if s <= 0.0:            # non-positive pivot -> not positive definite
                    return None
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def is_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> bool:
    """True iff A is symmetric within sym_tol and positive definite."""
    return cholesky_spd(A, sym_tol) is not None
