import numpy as np


def _oracle(X):
    X = np.asarray(X, dtype=np.float64)
    scales = np.max(np.abs(X), axis=1, keepdims=True) / 127.0
    q = np.zeros_like(X, dtype=np.int8)
    np.divide(X, scales, out=np.zeros_like(X), where=scales != 0)
    q = np.rint(np.divide(X, scales, out=np.zeros_like(X), where=scales != 0)).astype(np.int8)
    return q.astype(np.float64) * scales


def grade(sol, fx) -> dict:
    cases = [
        np.array([
            [512.0, -2.0, 1.0, 0.5],
            [0.25, -0.5, 0.1, 0.05],
            [3.0, -4.0, 1.0, 2.0],
        ]),
        np.array([
            [1000.0, 4.0, -3.0, 2.0],
            [8.0, -7.0, 1.0, 0.5],
        ]),
        np.array([
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.25],
            [-25.0, 4.0, 8.0],
        ]),
    ]

    worst = 0.0
    try:
        for X in cases:
            got = np.asarray(sol.per_token_int8_dequant(X), dtype=np.float64)
            ref = _oracle(X)
            if got.shape != ref.shape:
                return {"max_abs_err": float("inf")}
            worst = max(worst, float(np.max(np.abs(got - ref))))
    except Exception:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": worst}
