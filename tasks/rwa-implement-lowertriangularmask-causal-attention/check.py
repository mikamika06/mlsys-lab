import numpy as np
from mlsys.scorers import max_abs_err

def _ref(logits):
    out = np.asarray(logits, dtype=np.float64).copy()
    upper = np.triu_indices_from(out, k=1)
    out[upper] = -np.inf
    return out

def grade(sol, fx) -> dict:
    cases = [
        np.random.randn(5, 5).tolist(),
        (np.random.rand(8, 8) * 10).tolist(),
        [[0, 1], [2, 3]],
        np.full((4, 4), 7.5, dtype=np.float32).tolist(),
    ]
    max_err = 0.0
    for logits in cases:
        try:
            got = sol.causal_mask(logits)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        ref = _ref(logits)
        got_arr = np.asarray(got)
        if got_arr.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = max_abs_err(ref, got_arr)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
