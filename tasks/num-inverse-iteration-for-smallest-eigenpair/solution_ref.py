"""Reference solution for `num-inverse-iteration-for-smallest-eigenpair`."""
from __future__ import annotations

import numpy as np


def inverse_iteration(A: np.ndarray, num_iters: int = 100, x0: np.ndarray | None = None):
    """Inverse iteration: converges to the eigenpair of smallest |eigenvalue|.

    Each step solves ``A y = x`` (instead of multiplying by ``A``, as plain
    power iteration would), then re-normalizes. Since solving by ``A`` is
    the same as multiplying by ``A^{-1}``, the iterate converges to the
    dominant eigenvector of ``A^{-1}``, i.e. the eigenvector of ``A`` whose
    eigenvalue has the smallest absolute value.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    x = np.full(n, 1.0 / np.sqrt(n)) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    x = x / (np.linalg.norm(x) + 1e-15)

    for _ in range(num_iters):
        y = np.linalg.solve(A, x)
        x = y / (np.linalg.norm(y) + 1e-15)

    eigval = float(x @ A @ x)   # Rayleigh quotient
    return eigval, x
