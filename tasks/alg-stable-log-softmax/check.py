import numpy as np
from mlsys.scorers import max_abs_err

def _reference(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    # Numerically stable log-softmax using NumPy primitives
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(logits - m)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    return logits - m - log_sum_exp

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    errors = []
    # Test 1D array
    for shape in [(10,), (5,)]:
        logits = rng.standard_normal(shape) * 1000.0
        ref = _reference(logits)
        try:
            cand = sol.stable_log_softmax(logits)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        errors.append(max_abs_err(ref, cand))
    # Test 2D array with different axes
    for shape in [(3,4), (5,2)]:
        logits = rng.standard_normal(shape) * 1000.0
        for axis in [0, -1]:
            ref = _reference(logits, axis=axis)
            try:
                cand = sol.stable_log_softmax(logits, axis=axis)
            except Exception as e:
                return {"max_abs_err": float("inf")}
            errors.append(max_abs_err(ref, cand))
    # Test 3D array
    logits = rng.standard_normal((4,5,6)) * 1000.0
    for axis in [0,1,2]:
        ref = _reference(logits, axis=axis)
        try:
            cand = sol.stable_log_softmax(logits, axis=axis)
        except Exception as e:
            return {"max_abs_err": float("inf")}
        errors.append(max_abs_err(ref, cand))
    max_error = max(errors) if errors else 0.0
    return {"max_abs_err": max_error}
