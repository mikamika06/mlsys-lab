import numpy as np


def _oracle(X, Q, scales, zeros, group_size):
    m, k = Q.shape
    W = np.empty((m, k), dtype=np.float64)
    for i in range(m):
        g = i // group_size
        W[i] = (Q[i].astype(np.float64) - float(zeros[g])) * float(scales[g])
    return X.astype(np.float64) @ W.T


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.5, -1.0, 2.0], [1.5, 0.0, -0.5]], dtype=np.float64),
            np.array([[2, 4, 6], [7, 5, 3], [8, 1, 9], [4, 6, 2]], dtype=np.int8),
            np.array([0.25, 0.5], dtype=np.float64),
            np.array([2, 1], dtype=np.int64),
            2,
        ),
        (
            np.array([[1.0, 2.0, 3.0, -1.0]], dtype=np.float64),
            np.array([[1, 3, 5, 7], [8, 6, 4, 2], [9, 9, 1, 0]], dtype=np.int8),
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0, 3], dtype=np.int64),
            2,
        ),
    ]

    worst = 0.0
    for X, Q, scales, zeros, group_size in cases:
        try:
            got = np.asarray(sol.quantized_linear(X, Q, scales, zeros, group_size), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(X, Q, scales, zeros, group_size)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
