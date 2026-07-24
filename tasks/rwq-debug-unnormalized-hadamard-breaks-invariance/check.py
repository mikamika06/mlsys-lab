import numpy as np


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -2.0], [3.0, 4.0]]),
            np.array([[0.5, 1.5], [-2.0, 3.0]]),
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4) / 7.0,
            np.array([[1.0], [2.0], [-1.0], [0.5]]),
        ),
        (
            np.arange(32, dtype=np.float64).reshape(4, 8) / 11.0,
            np.arange(24, dtype=np.float64).reshape(8, 3) / 13.0,
        ),
    ]

    worst = 0.0
    for X, W in cases:
        n = X.shape[1]
        H = _hadamard(n)
        Q = H / np.sqrt(n)
        ref = X @ W
        try:
            X_rot, W_rot = sol.hadamard_rotate(X.copy(), W.copy())
            got = np.asarray(X_rot, dtype=np.float64) @ np.asarray(W_rot, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
