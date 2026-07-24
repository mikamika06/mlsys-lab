import numpy as np


def _reference(x):
    return np.cumsum(np.asarray(x, dtype=np.float64), dtype=np.float64)


def grade(sol, fx) -> dict:
    cases = [
        np.array([1.0, 2.0, 3.0]),
        np.array([-5.0, 1.5, 0.25, 8.0]),
        np.array([0.0]),
        np.array([1e-10, 1e10, -1e10, 3.0]),
        np.linspace(-3.0, 3.0, 101),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = np.asarray(sol.inclusive_scan(x), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _reference(x)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
