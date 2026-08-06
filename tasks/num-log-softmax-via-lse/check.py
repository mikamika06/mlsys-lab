import numpy as np

def _ref_log_softmax(x):
    """Stable log-softmax via the log-sum-exp trick (the oracle)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return x - (m + np.log(np.sum(np.exp(x - m))))

def grade(sol, fx) -> dict:
    """Compute worst-case max_abs_err across all test vectors."""
    cases = [
        [0.0, 1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0],
        [-1000.0, -1001.0, -1002.0],
        [-500.0, 0.0, 500.0],
        [1.0, 1.0, 1.0],
        [0.0],
        [-1e6, 0.0, 1e6],
    ]
    worst = 0.0
    for x in cases:
        try:
            got = sol.log_softmax(list(x))
            ref = _ref_log_softmax(np.asarray(x, dtype=np.float64))
        except Exception:
            return {"max_abs_err": float("inf")}
        if not isinstance(got, list) or len(got) != len(ref):
            return {"max_abs_err": float("inf")}
        got_arr = np.asarray(got, dtype=np.float64)
        err = float(np.max(np.abs(got_arr - ref)))
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)
    return {"max_abs_err": worst}
