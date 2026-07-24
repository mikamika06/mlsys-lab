import numpy as np


def _oracle(q, K, V):
    scores = K @ q
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)
    return (weights[:, None] * V).sum(axis=0)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 0.0]]),
            np.array([[2.0, 1.0], [4.0, 2.0], [8.0, 3.0]]),
            1,
        ),
        (
            np.array([0.5, -1.0, 2.0]),
            np.array([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [4.0, -1.0, 2.0],
                [0.2, 0.1, 0.9],
                [5.0, 0.0, 1.0],
            ]),
            np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
                [6.0, 7.0],
            ]),
            2,
        ),
        (
            np.array([-2.0, 1.5]),
            np.array([
                [1.0, 2.0],
                [3.0, -1.0],
                [6.0, 4.0],
                [-1.0, 0.5],
            ]),
            np.array([
                [1.0],
                [3.0],
                [2.0],
                [9.0],
            ]),
            3,
        ),
    ]

    worst = 0.0
    for q, K, V, block_size in cases:
        try:
            got = np.asarray(sol.flash_attention_accumulate(q, K, V, block_size), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(q.astype(np.float64), K.astype(np.float64), V.astype(np.float64))
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
