import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tests = [
        rng.normal(size=(3, 4)),
        rng.uniform(-10, 10, size=(5, 6)),
        rng.uniform(-1000, 1000, size=(2, 8)),
        rng.standard_normal(size=(7, 5)) * 200,
        rng.exponential(scale=1e3, size=(4, 9)) - 500,
    ]
    max_err = 0.0
    for x in tests:
        try:
            cand = sol.log_softmax(x)
        except Exception:
            return {"max_abs_err": float("inf")}
        mx = np.max(x, axis=-1, keepdims=True)
        ref = -mx + np.log(np.sum(np.exp(x - mx), axis=-1, keepdims=True))
        err = max_abs_err(ref, cand)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
