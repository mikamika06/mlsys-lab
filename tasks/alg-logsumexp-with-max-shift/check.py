import numpy as np

def _reference_logsumexp(x, axis):
    if axis is None:
        max_val = np.max(x)
        return np.log(np.sum(np.exp(x - max_val))) + max_val
    else:
        max_val = np.max(x, axis=axis, keepdims=True)
        sum_exp = np.sum(np.exp(x - max_val), axis=axis)
        return np.squeeze(np.log(sum_exp) + np.squeeze(max_val))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (np.array([-1., 0., 1.]), None),
        (np.array([1000., 1001., 1002.]), None),
        (np.array([-1000., -999., -998.]), None),
        (rng.normal(size=(5, 4)), 0),
        (rng.uniform(-10, 10, size=(3,)), None),
        (np.array([[1., 2.], [3., 4.]]), 1)
    ]

    max_err = 0.0
    for x, axis in cases:
        try:
            got = sol.logsumexp(x, axis=axis)
            ref = _reference_logsumexp(np.asarray(x, dtype=np.float64), axis)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = np.max(np.abs(got.astype(np.float64) - ref))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
