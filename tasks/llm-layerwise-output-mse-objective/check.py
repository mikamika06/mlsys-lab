import numpy as np


def _oracle(W, W_q, X):
    W = np.asarray(W, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = W @ X
    y_q = W_q @ X
    return float(np.mean((y - y_q) ** 2))


def grade(sol, fx) -> dict:
    cases = [
        (
            [[1.0, -2.0], [3.0, 4.0]],
            [[1.0, -1.5], [2.5, 4.0]],
            [[1.0, 0.0, 2.0], [0.5, -1.0, 1.5]],
        ),
        (
            np.arange(12, dtype=np.float64).reshape(3, 4).tolist(),
            (np.arange(12, dtype=np.float64).reshape(3, 4) + 0.25).tolist(),
            [
                [1.0, -1.0],
                [0.5, 2.0],
                [3.0, 0.0],
                [-2.0, 1.0],
            ],
        ),
        (
            [[0.1, -0.2, 0.3]],
            [[0.0, -0.25, 0.25]],
            [[2.0, 1.0], [-1.0, 0.5], [0.0, 3.0]],
        ),
    ]

    errors = []
    for W, W_q, X in cases:
        try:
            got = float(sol.layerwise_output_mse(W, W_q, X))
        except Exception:
            return {"mse": float("inf")}
        ref = _oracle(W, W_q, X)
        errors.append((got - ref) ** 2)

    return {"mse": float(np.mean(errors))}
