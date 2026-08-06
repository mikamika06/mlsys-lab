import numpy as np
from mlsys.scorers import max_abs_err

def _reference(z):
    z_arr = np.asarray(z, dtype=np.float64)
    return np.where(
        z_arr >= 0,
        1.0 / (1.0 + np.exp(-z_arr)),
        np.exp(z_arr) / (1.0 + np.exp(z_arr))
    )

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    tests = [
        [-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0],
        rng.uniform(-1000, 1000, size=(20,)).tolist(),
        (rng.standard_normal(size=1000) * 500).tolist(),
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
