import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    err_max = 0.0
    for _ in range(5):
        d = rng.integers(1, 20)
        a = rng.standard_normal(d).astype(np.float64)
        b = rng.standard_normal(d).astype(np.float64)
        try:
            got = sol.sq_dist_expansion(a, b)
        except Exception:
            return {"max_abs_err": float("inf")}
        naive = np.sum((a - b) ** 2)
        err = scorers.max_abs_err(np.array([got]), np.array([naive]))
        if err > err_max:
            err_max = err
    return {"max_abs_err": err_max}
