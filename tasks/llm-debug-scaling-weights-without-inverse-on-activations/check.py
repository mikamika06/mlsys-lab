import numpy as np


def _oracle(X, W, s):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    W_scaled = W * s[None, :]
    X_compensated = X / s[None, :]
    return X_compensated @ W_scaled.T


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0], [3.0, -1.0]]),
            np.array([[3.0, 4.0], [-2.0, 5.0]]),
            np.array([2.0, 5.0]),
        ),
        (
            np.array([[0.5, -3.0, 4.0], [2.0, 1.0, -1.0]]),
            np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]]),
            np.array([0.5, 3.0, 7.0]),
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4) / 3.0,
            np.arange(20, dtype=np.float64).reshape(5, 4) / 7.0,
            np.array([1.5, 2.0, 0.25, 4.0]),
        ),
    ]

    worst = 0.0
    for X, W, s in cases:
        try:
            got = np.asarray(sol.restore_awq_equivalence(X, W, s), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(X, W, s)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
