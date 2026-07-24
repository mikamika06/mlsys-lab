import numpy as np

def _reference(x):
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))

def grade(sol, fx) -> dict:
    tests = [
        np.array([-1000, -1, 0, 1, 1000], dtype=np.float64),
        np.random.default_rng(0).uniform(-500, 500, size=(10,)).astype(np.float64),
        np.linspace(-500, 500, num=20, dtype=np.float64),
    ]
    max_err = 0.0
    for x in tests:
        try:
            cand = sol.softplus(x)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference(x)
        err = np.max(np.abs(cand.astype(np.float64) - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
