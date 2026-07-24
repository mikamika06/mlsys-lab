"""Grader for `num-inverse-iteration-for-smallest-eigenpair`.

Oracle: `np.linalg.eigh` (LAPACK) on deterministic, well-separated
symmetric positive-definite test matrices. Never hardcoded.
"""
from __future__ import annotations

import numpy as np


def _spd_case(rng, n):
    """SPD matrix whose smallest eigenvalue is well separated from the
    second-smallest, so inverse iteration converges fast and reliably."""
    lam = np.sort(rng.uniform(1.0, 6.0, size=n))
    lam[0] = lam[1] * 0.3          # force a clear gap at the bottom
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(lam) @ Q.T
    return 0.5 * (A + A.T)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    rel_errs = []
    vec_errs = []

    for _ in range(6):
        n = int(rng.integers(4, 9))
        A = _spd_case(rng, n)

        eigvals, eigvecs = np.linalg.eigh(A)
        true_val = float(eigvals[0])
        true_vec = eigvecs[:, 0]

        try:
            val, vec = sol.inverse_iteration(A.copy(), num_iters=100)
            val = float(val)
            vec = np.asarray(vec, dtype=np.float64).ravel()
        except Exception:
            return {"rel_err": float("inf"), "vec_err": float("inf")}

        if vec.shape != (n,) or not np.isfinite(val) or not np.all(np.isfinite(vec)):
            return {"rel_err": float("inf"), "vec_err": float("inf")}

        rel_errs.append(abs(val - true_val) / (abs(true_val) + 1e-12))

        vnorm = np.linalg.norm(vec)
        if vnorm < 1e-12:
            vec_errs.append(1.0)
            continue
        vn = vec / vnorm
        cos = abs(float(np.dot(vn, true_vec)))     # sign of eigenvector is arbitrary
        vec_errs.append(1.0 - cos)

    return {"rel_err": float(np.mean(rel_errs)), "vec_err": float(np.mean(vec_errs))}
