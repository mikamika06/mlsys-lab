from __future__ import annotations

import numpy as np

SYM_TOL = 1e-10


def cholesky_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> np.ndarray | None:
    """Return lower-triangular L with L @ L.T == A, or None if A is not SPD.

    A is not SPD if it is non-square, not symmetric within `sym_tol`, or the
    factorisation hits a non-positive pivot.
    """
    raise NotImplementedError('your code here')


def is_spd(A: np.ndarray, sym_tol: float = SYM_TOL) -> bool:
    """Return True iff A is symmetric within `sym_tol` and positive definite."""
    raise NotImplementedError('your code here')
