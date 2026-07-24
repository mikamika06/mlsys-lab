"""Grader for `num-fix-cholesky-missing-diagonal-sqrt`.

Two independent oracles, both computed here, never hardcoded:
  * the factorisation identity  A = L L^T  (max_abs_err of the reconstruction);
  * LAPACK's own factor via `np.linalg.cholesky`, which is unique for an SPD
    matrix once the diagonal is required positive.
"""
from __future__ import annotations

import numpy as np

from mlsys import scorers


def _spd_cases():
    """Deterministic, well-conditioned symmetric positive definite matrices."""
    out = []
    for seed, n in [(0, 4), (1, 7), (2, 12), (3, 25)]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n, n))
        A = W @ W.T + n * np.eye(n)
        A = 0.5 * (A + A.T)            # kill any asymmetry from rounding
        out.append(np.ascontiguousarray(A, dtype=np.float64))
    return out


def grade(sol, fx) -> dict:
    recon_err = 0.0
    oracle_err = 0.0

    for A in _spd_cases():
        try:
            L = np.asarray(sol.cholesky(A.copy()), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "oracle_max_abs_err": float("inf")}

        if L.shape != A.shape or not np.all(np.isfinite(L)):
            return {"max_abs_err": float("inf"), "oracle_max_abs_err": float("inf")}

        recon_err = max(recon_err, scorers.max_abs_err(A, L @ L.T))
        oracle_err = max(oracle_err, scorers.max_abs_err(np.linalg.cholesky(A), L))

    return {"max_abs_err": float(recon_err), "oracle_max_abs_err": float(oracle_err)}
