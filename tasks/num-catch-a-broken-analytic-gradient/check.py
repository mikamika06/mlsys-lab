import numpy as np


def _loss(x):
    x = np.asarray(x, dtype=np.float64)
    i = np.arange(x.size, dtype=np.float64)
    return float(np.sum((i + 1) * x**3 + 2 * x**2 - 5 * x))


def _finite_diff_grad(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        out[i] = (_loss(xp) - _loss(xm)) / (2 * eps)
    return out


def grade(sol, fx) -> dict:
    cases = [
        np.array([1.0, -2.0, 0.5]),
        np.array([-1.5, 0.25, 3.0, -0.75]),
        np.array([0.0, 2.0, -1.0, 1.5, -2.5]),
    ]
    max_err = 0.0
    for x in cases:
        try:
            got = np.asarray(sol.fixed_gradient(x), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _finite_diff_grad(x)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": max_err}
