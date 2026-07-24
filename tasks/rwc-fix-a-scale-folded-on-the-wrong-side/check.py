import numpy as np
from mlsys import scorers


def _oracle(W, X, s):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    W_scaled = W * s.reshape(1, -1)
    X_fixed = X / s.reshape(1, -1)
    return X_fixed @ W_scaled.T


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.8, -1.2, 2.0], [1.5, 0.3, -0.7]], dtype=np.float64),
            np.array([[1.0, 2.0, -1.0], [0.5, -0.4, 3.0]], dtype=np.float64),
            np.array([2.0, 0.5, 3.0], dtype=np.float64),
        ),
        (
            np.array([[3.0, -2.0], [1.0, 4.0], [-1.0, 0.5]], dtype=np.float64),
            np.array([[2.0, -1.0]], dtype=np.float64),
            np.array([4.0, 0.25], dtype=np.float64),
        ),
    ]

    worst = 0.0
    for W, X, s in cases:
        ref = _oracle(W, X, s)
        try:
            got = sol.fix_awq_scale(W, X, s)
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, scorers.max_abs_err(ref, got))

        broken = X @ (W * s.reshape(1, -1)).T
        if scorers.max_abs_err(ref, broken) < 1e-8:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": worst}
