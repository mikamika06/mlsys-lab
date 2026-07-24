import numpy as np


def _oracle(q, K_blocks, V_blocks):
    K = np.concatenate(K_blocks, axis=0).astype(np.float64)
    V = np.concatenate(V_blocks, axis=0).astype(np.float64)
    q = np.asarray(q, dtype=np.float64)
    scores = K @ q / np.sqrt(float(q.shape[0]))
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)
    return weights @ V


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, -0.5, 2.0]),
            [
                np.array([[0.2, 1.0, -0.3], [1.5, -0.2, 0.7]]),
                np.array([[0.1, 0.4, 2.0]]),
            ],
            [
                np.array([[1.0, 2.0], [3.0, -1.0]]),
                np.array([[4.0, 0.5]]),
            ],
        ),
        (
            np.array([3.0, 2.0]),
            [
                np.array([[10.0, 0.0]]),
                np.array([[0.0, 10.0], [5.0, 5.0]]),
                np.array([[1.0, -1.0]]),
            ],
            [
                np.array([[5.0]]),
                np.array([[7.0], [9.0]]),
                np.array([[2.0]]),
            ],
        ),
    ]

    worst = 0.0
    for q, kb, vb in cases:
        try:
            got = np.asarray(sol.online_attention(q, kb, vb), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(q, kb, vb)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
