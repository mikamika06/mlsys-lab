import numpy as np
from mlsys import scorers

def _reference(X, W, alpha):
    max_X = np.max(np.abs(X), axis=0)
    max_W = np.max(np.abs(W), axis=0)
    return (max_X ** alpha) / (max_W ** (1 - alpha))

def grade(sol, fx) -> dict:
    # Generate random test cases
    rng = np.random.default_rng(42)
    rel_errs = []
    for _ in range(5):
        n, d = rng.integers(2, 10), rng.integers(3, 8)
        X = rng.standard_normal((n, d))
        W = rng.standard_normal((n, d))
        alpha = rng.random()
        try:
            out = sol.per_channel_scale(X, W, alpha)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(X, W, alpha)
        rel_errs.append(scorers.rel_err(ref, out))
    # Use the worst relative error
    max_rel_err = max(rel_errs)
    return {"rel_err": max_rel_err}
