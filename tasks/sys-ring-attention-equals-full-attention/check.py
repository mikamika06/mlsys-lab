import numpy as np


def _full_attention(Q, K, V):
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ V


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]]),
            np.array([[2.0, 1.0], [3.0, 4.0], [5.0, 6.0]]),
            2,
        ),
        (
            np.arange(24, dtype=np.float64).reshape(4, 6) / 10.0,
            np.flip(np.arange(24, dtype=np.float64).reshape(4, 6), axis=0) / 11.0,
            np.arange(16, dtype=np.float64).reshape(4, 4) / 7.0,
            4,
        ),
        (
            np.array([[0.2, -0.1, 0.4], [0.3, 0.8, -0.5]]),
            np.array([[0.7, -0.2, 0.1], [-0.4, 0.6, 0.9]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
        ),
    ]

    worst = 0.0
    for Q, K, V, ranks in cases:
        try:
            got = np.asarray(sol.ring_attention(Q, K, V, ranks), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _full_attention(Q, K, V)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
