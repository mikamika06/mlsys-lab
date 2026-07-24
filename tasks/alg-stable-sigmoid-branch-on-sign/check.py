import numpy as np
from mlsys.scorers import max_abs_err

def _reference(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    tests = [
        np.array([-1000., -10., -1., 0., 1., 10., 1000.]),
        rng.uniform(-1000, 1000, size=(5, 4)),
        rng.standard_normal(size=10000) * 500,
    ]
    max_err = 0.0
    for z in tests:
        try:
            got = sol.stable_sigmoid(z)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference(z)
        err = max_abs_err(ref, got)
        if err > max_err:
            max_err = err
        if err > 1e-12:
            return {"max_abs_err": err}
    return {"max_abs_err": max_err}
