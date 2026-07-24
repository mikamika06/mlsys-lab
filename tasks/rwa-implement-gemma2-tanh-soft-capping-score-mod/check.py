import numpy as np


def _oracle(Q, K, V, cap):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d)
    scores = cap * np.tanh(scores / cap)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 0.0], [0.5, -1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
            np.array([[2.0, 1.0], [0.0, 3.0], [4.0, -1.0]]),
            1.0,
        ),
        (
            np.array([[2.0, -1.0, 0.5]]),
            np.array([[1.0, 0.0, 2.0], [-1.0, 1.0, 0.5]]),
            np.array([[1.0], [5.0]]),
            10.0,
        ),
        (
            np.array([[0.2, 0.4], [-0.3, 0.8], [1.0, -1.0]]),
            np.array([[0.5, 0.5], [-0.5, 1.5], [2.0, -1.0]]),
            np.array([[1.0, 2.0], [3.0, 0.0], [-2.0, 4.0]]),
            2.0,
        ),
    ]

    max_err = 0.0
    for Q, K, V, cap in cases:
        try:
            got = sol.attention_with_score_mod(Q, K, V, cap)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(Q, K, V, cap)
        err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
