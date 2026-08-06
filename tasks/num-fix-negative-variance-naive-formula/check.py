import numpy as np


def _oracle(x):
    return float(np.var(np.asarray(x, dtype=np.float64)))


def grade(sol, fx) -> dict:
    cases = [
        [1e12, 1e12 + 1, 1e12 - 1],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [-5.5, 2.25, 7.75, 11.0],
        list(np.linspace(1e9, 1e9 + 100, 1000, dtype=float)),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = float(sol.stable_variance(list(x)))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(x)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
