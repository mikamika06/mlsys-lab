import numpy as np

def _stable_log_softmax(x):
    """Reference log-softmax computed with the stable LSE identity."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        m = np.max(x)
        lse = m + np.log(np.sum(np.exp(x - m)))
        return x - lse
    else:
        m = np.max(x, axis=-1, keepdims=True)
        lse = m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))
        return x - lse

def grade(sol, fx) -> dict:
    cases = [
        [0.0, 1.0, 2.0, 3.0],
        [1000.0, 1001.0, 1002.0, 1003.0],
        [-1000.0, -1001.0, -1002.0, -1003.0],
        [-100.0, 0.0, 100.0],
        [42.0],
        [5.0, 5.0, 5.0, 5.0],
        np.random.RandomState(42).uniform(-50, 50, (8, 12)).tolist(),
        np.random.RandomState(99).uniform(-500, 500, (16, 6)).tolist(),
        np.linspace(-200, 200, 32).tolist(),
        [1e6, 1e6 - 1, 1e6 - 2],
    ]
    worst = 0.0
    for x in cases:
        try:
            got = np.asarray(sol.log_softmax(x), dtype=np.float64)
            ref = _stable_log_softmax(x)
        except Exception:
            return {"max_abs_err": 1e30}
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)
    return {"max_abs_err": worst}
