import numpy as np


def _oracle(x):
    x = np.asarray(x, dtype=np.longdouble)
    half = x / np.longdouble(2)
    return np.asarray(0.5 * (np.sin(half) / half) ** 2, dtype=np.float64)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        np.array([1e-2, 1e-4, 1e-8, 1e-12], dtype=np.float64),
        np.array([-1e-3, -1e-7, 3e-9, 7e-13], dtype=np.float64),
        np.array([5e-5, 2e-10, -4e-11, 9e-14], dtype=np.float64),
    ]

    worst = 0.0
    for x in cases:
        try:
            got = sol.stable_one_minus_cos_over_x2(x)
        except Exception:
            return {"rel_err": float("inf")}

        err = _rel_err(got, _oracle(x))
        worst = max(worst, err)

    return {"rel_err": worst}
