import numpy as np


def _oracle_softmax_vjp(x, g):
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    shifted = x - np.max(x)
    e = np.exp(shifted)
    s = e / np.sum(e)
    return s * (g - np.sum(g * s))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, -1.0, 2.0]),
        ),
        (
            np.array([-4.0, 0.0, 4.0, 8.0]),
            np.array([1.0, 2.0, -3.0, 0.25]),
        ),
        (
            np.array([0.2, -0.7, 1.3, 2.1, -3.0]),
            np.array([-1.5, 0.4, 0.8, -0.2, 2.0]),
        ),
    ]

    worst = 0.0
    for x, g in cases:
        try:
            got = np.asarray(sol.softmax_vjp(x, g), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle_softmax_vjp(x, g)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
