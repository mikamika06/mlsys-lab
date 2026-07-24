import numpy as np
from mlsys.scorers import max_abs_err

def _reference(x, beta, p):
    return np.sign(x) * np.maximum(np.abs(x) - beta * np.power(np.abs(x), p-1), 0.0)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    errors = []
    for n in [10, 20, 50]:
        x = rng.standard_normal(n).astype(np.float64)
        beta = rng.uniform(0.1, 2.0)
        p = 0.7
        try:
            out = sol.generalized_soft_threshold(x, beta, p)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference(x, beta, p)
        errors.append(max_abs_err(ref, out))
    max_error = max(errors)
    return {"max_abs_err": max_error}
