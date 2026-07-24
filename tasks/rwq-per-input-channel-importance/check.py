import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    # Deterministic tensor for reproducibility
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 5, 6), dtype=np.float64)
    ref = np.mean(np.abs(X), axis=(0, 1))
    try:
        got = sol.per_input_channel_importance(X)
        got_arr = np.asarray(got, dtype=np.float64)
        ref_arr = np.asarray(ref, dtype=np.float64)
        err = scorers.rel_err(ref_arr, got_arr)
    except Exception:
        return {"rel_err": float("inf")}
    return {"rel_err": err}
