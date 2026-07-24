import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    try:
        func = getattr(sol, "log1p_tiny")
    except AttributeError:
        return {"rel_err": float("inf")}
    xs = np.array(
        [1e-16, 2e-15, 5e-14, 1e-13, 3e-12, 7e-11, 2e-10, 5e-9, 8e-9, 1e-8],
        dtype=np.float64,
    )
    ref = np.log1p(xs)
    try:
        got = func(xs)
    except Exception:
        return {"rel_err": float("inf")}
    err = rel_err(ref, got)
    return {"rel_err": err}
