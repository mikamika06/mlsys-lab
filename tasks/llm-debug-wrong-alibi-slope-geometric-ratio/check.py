import numpy as np


def _oracle_alibi_slopes(n):
    h = np.arange(n, dtype=np.float64)
    return np.power(2.0, -8.0 * h / n)


def grade(sol, fx) -> dict:
    max_err = 0.0
    for n in [1, 2, 4, 8, 12, 32, 64]:
        try:
            got = np.asarray(sol.alibi_slopes(n), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle_alibi_slopes(n)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": max_err}
