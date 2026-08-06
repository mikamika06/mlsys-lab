import numpy as np

def _build_test_cases():
    """Build test data in-memory with logits up to +1000."""
    rng = np.random.default_rng(42)
    row1 = np.array([1000.0, 999.0, 998.0, 997.0, 0.0, -10.0, -50.0])
    row2 = np.full(7, 1000.0)
    row3 = np.full(7, -1000.0)
    row4 = np.array([1000.0, -1000.0, 500.0, -500.0, 0.0, 200.0, -200.0])
    batch = rng.uniform(-1000, 1000, size=(20, 7))
    return np.vstack([row1, row2, row3, row4, batch])  # (24, 7)

def _ref_stable_softmax(logits):
    """Oracle reference: subtract max, exponentiate, normalize."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)

def grade(sol, fx) -> dict:
    logits = _build_test_cases()

    try:
        got = np.asarray(sol.stable_softmax(logits.tolist()), dtype=np.float64)
    except Exception:
        return {"max_abs_err": float("inf")}

    # Any inf or nan in the output is an immediate failure
    if np.any(np.isnan(got)) or np.any(np.isinf(got)):
        return {"max_abs_err": float("inf")}

    ref = _ref_stable_softmax(logits)

    if got.shape != ref.shape:
        return {"max_abs_err": float("inf")}

    err = float(np.max(np.abs(got - ref)))
    return {"max_abs_err": err}
