import numpy as np


def _oracle(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    d = q.shape[1]
    logits = (q @ k.T) / np.sqrt(d)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights @ v


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        ),
        (
            np.array([[0.2, -1.0, 0.5, 2.0], [1.0, 1.0, -0.5, 0.3]]),
            np.array([
                [0.5, 0.2, -0.1, 1.0],
                [-0.7, 0.4, 0.8, -0.2],
                [1.1, -0.3, 0.2, 0.5],
            ]),
            np.array([
                [1.0, 2.0],
                [3.0, -1.0],
                [0.5, 0.5],
            ]),
        ),
        (
            np.arange(15, dtype=np.float64).reshape(3, 5) / 10.0,
            np.arange(20, dtype=np.float64).reshape(4, 5) / 7.0,
            np.arange(12, dtype=np.float64).reshape(4, 3) / 5.0,
        ),
    ]

    worst = 0.0
    for q, k, v in cases:
        try:
            got = np.asarray(sol.scaled_dot_product_attention(q, k, v), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(q, k, v)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
