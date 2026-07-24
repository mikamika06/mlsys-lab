import numpy as np
from mlsys.scorers import max_abs_err

def _ref(x):
    """Reference RMSNorm using float64 reduction."""
    y = x.astype(np.float64)
    mean_sq = np.mean(y * y, axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq)
    out = (x / rms).astype(np.float16)
    return out

def grade(sol, fx) -> dict:
    # Generate a few random test cases
    rng = np.random.default_rng(0)
    shapes = [(10, 128), (32, 256), (64, 512), (5, 3)]
    max_err = 0.0
    for shape in shapes:
        x = rng.standard_normal(shape).astype(np.float16)
        try:
            cand = sol.rmsnorm(x, upcast=True)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _ref(x)
        err = max_abs_err(ref, cand)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
