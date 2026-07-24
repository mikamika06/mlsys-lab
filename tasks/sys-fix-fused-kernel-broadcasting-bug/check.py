import numpy as np


def _oracle(X, scale, bias):
    return np.asarray(X, dtype=np.float64) * np.asarray(scale, dtype=np.float64) + np.asarray(bias, dtype=np.float64)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([2.0, 3.0, 4.0]),
            np.array([1.0, 0.0, -1.0]),
        ),
        (
            np.arange(24, dtype=np.float64).reshape(4, 6) - 3.0,
            np.array([1.5, -2.0, 0.5, 3.0, -1.0, 2.5]),
            np.array([0.25, 1.0, -0.5, 2.0, 3.0, -4.0]),
        ),
        (
            np.array([[10.0, -2.0], [0.5, 8.0], [-3.0, 7.0]]),
            np.array([-1.0, 4.0]),
            np.array([5.0, -2.0]),
        ),
    ]

    worst = 0.0
    for X, scale, bias in cases:
        try:
            got = sol.fused_affine(X, scale, bias)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(X, scale, bias)
        err = _rel_err(got, ref)
        worst = max(worst, err)
    return {"rel_err": worst}
