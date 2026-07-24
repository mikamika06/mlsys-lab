import numpy as np
from mlsys import scorers

def _reference(logits, T):
    logits = np.asarray(logits, dtype=np.float64)
    z = logits / T
    shift = np.max(z)
    exp_z = np.exp(z - shift)
    return exp_z / np.sum(exp_z)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    # random cases
    for _ in range(9):
        size = rng.integers(5, 30)
        logits = rng.standard_normal(size).astype(np.float64) * 10.0
        T = rng.uniform(0.01, 5.0)
        try:
            got = sol.softmax_temperature(logits, T)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, np.ndarray):
            return {"rel_err": float("inf")}
        ref = _reference(logits, T)
        err = scorers.rel_err(ref, got)
        if not np.isfinite(err):
            return {"rel_err": float("inf")}
        if err > max_err:
            max_err = err
    # extreme case to catch naive overflow
    logits = np.array([1000.0, -1000.0, 500.0, -500.0, 0.0], dtype=np.float64)
    T = 0.01
    try:
        got = sol.softmax_temperature(logits, T)
    except Exception:
        return {"rel_err": float("inf")}
    if not isinstance(got, np.ndarray):
        return {"rel_err": float("inf")}
    ref = _reference(logits, T)
    err = scorers.rel_err(ref, got)
    if not np.isfinite(err):
        return {"rel_err": float("inf")}
    if err > max_err:
        max_err = err
    return {"rel_err": max_err}
