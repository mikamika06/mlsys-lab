import numpy as np


def _reference(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def grade(sol, fx) -> dict:
    cases = [
        np.array([-1000.0, -100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0, 1000.0]),
        np.linspace(-1000.0, 1000.0, 1001),
        np.array([[ -50.5, 0.25, 3.0 ], [700.0, -700.0, 1e-12]]),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = np.asarray(sol.stable_softplus(x), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _reference(x)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
