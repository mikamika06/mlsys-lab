import numpy as np
from mlsys.scorers import channel_rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    # Define a few representative shapes
    cases = [
        (rng.standard_normal((10, 3)), "small"),
        (rng.standard_normal((1000, 64)), "medium"),
        (rng.standard_normal((500, 256)), "large")
    ]
    max_err = 0.0
    for X, _ in cases:
        try:
            got = sol.per_channel_scales(X)
            if not isinstance(got, np.ndarray):
                return {"channel_rel_err": float("inf")}
            ref = np.linalg.norm(X, axis=0) / np.sqrt(X.shape[0])
            err = channel_rel_err(ref, got, axis=0)
        except Exception:
            return {"channel_rel_err": float("inf")}
        if err > max_err:
            max_err = err
    return {"channel_rel_err": max_err}
