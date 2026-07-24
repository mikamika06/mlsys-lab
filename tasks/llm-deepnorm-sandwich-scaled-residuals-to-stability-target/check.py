import numpy as np
from mlsys import scorers

def _ref(residuals, alpha):
    norms = []
    for i, r in enumerate(residuals):
        norm = np.linalg.norm(r.astype(np.float64), ord=2)
        scaled = norm * (alpha ** i)
        norms.append(scaled)
    return np.array(norms, dtype=np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(10):
        L = rng.integers(1, 11)
        residuals = [rng.standard_normal(rng.integers(3,9)) for __ in range(L)]
        alpha = rng.uniform(0.5, 2.0)
        try:
            got = sol.deepnorm_scaled_residuals(residuals, float(alpha))
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref(residuals, float(alpha))
        err = scorers.rel_err(ref, got)
        errors.append(err)
    rel_err_val = float(np.mean(errors))
    return {"rel_err": rel_err_val}
