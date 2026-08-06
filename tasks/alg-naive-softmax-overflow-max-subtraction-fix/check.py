import numpy as np
from mlsys import scorers

def _reference(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - shift)
    return exps / np.sum(exps, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    cases = [
        np.array([0.0, 1.0, 2.0]),
        np.array([-1000.0, 0.0, 1000.0]),
        np.random.randn(5),
        np.random.randn(3, 4),
        np.full((2, 3), 12345.0),
    ]
    max_err = 0.0
    for arr in cases:
        logits = arr.tolist()
        try:
            got = sol.softmax(logits)
        except Exception:
            return {"max_abs_err": float("inf")}
        if not isinstance(got, list):
            return {"max_abs_err": float("inf")}
        expected = _reference(arr)
        err = scorers.max_abs_err(expected, np.asarray(got, dtype=np.float64))
        if np.isnan(err) or np.isinf(err):
            err = float("inf")
        max_err = max(max_err, err)
    return {"max_abs_err": max_err}
