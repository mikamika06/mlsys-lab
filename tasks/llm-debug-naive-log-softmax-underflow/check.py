import math
import numpy as np

def _stable_log_softmax(x):
    """Reference: numerically stable log-softmax via log-sum-exp trick."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True)) + x_max
    return x - lse

def grade(sol, fx) -> dict:
    cases = [
        [1.0, 2.0, 3.0],
        [-1000.0, -1001.0, -1002.0],
        [-500.0, 0.0, 1.0],
        [[-100.0, -200.0, -300.0], [1.0, 2.0, 3.0]],
        [-700.0, -701.0, -702.0, -703.0],
    ]

    max_err = 0.0
    all_finite = 1.0

    for x in cases:
        ref = _stable_log_softmax(x)
        try:
            got = np.asarray(sol.log_softmax(x), dtype=np.float64)
        except Exception:
            return {"max_abs_err": 1.0, "all_finite": 0.0}

        if not np.all(np.isfinite(got)):
            all_finite = 0.0

        err = float(np.max(np.abs(got - ref)))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err, "all_finite": all_finite}
