import numpy as np

def _ref_log_softmax(x):
    """Stable log-softmax via the log-sum-exp trick (the oracle)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return x - (m + np.log(np.sum(np.exp(x - m))))

def grade(sol, fx) -> dict:
    """Compute worst-case max_abs_err across all test vectors."""
    cases = [
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1000.0, 1001.0, 1002.0]),
        np.array([-1000.0, -1001.0, -1002.0]),
        np.array([-500.0, 0.0, 500.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0]),
        np.array([-1e6, 0.0, 1e6]),
    ]
    worst = 0.0
    for x in cases:
        x64 = x.astype(np.float64)
        try:
            got = np.asarray(sol.log_softmax(x64), dtype=np.float64)
            ref = _ref_log_softmax(x64)
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
