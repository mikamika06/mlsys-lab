import numpy as np


def _oracle(x):
    return float(np.var(np.asarray(x, dtype=np.float64)))


def grade(sol, fx) -> dict:
    cases = [
        np.array([1e12, 1e12 + 1, 1e12 - 1], dtype=np.float64),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.array([-5.5, 2.25, 7.75, 11.0], dtype=np.float64),
        np.linspace(1e9, 1e9 + 100, 1000, dtype=np.float64),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = float(sol.stable_variance(x.copy()))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(x)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
