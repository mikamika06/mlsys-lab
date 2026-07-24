"""Buggy Cholesky factorisation — find and fix the bug.

This "works" in the sense that it runs without error and produces a
lower-triangular matrix, but the reconstruction A = L @ L.T does not hold.
"""
from __future__ import annotations

import numpy as np


def cholesky(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                L[i, j] = A[i, i] - s          # BUG: forgot np.sqrt(...)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L
