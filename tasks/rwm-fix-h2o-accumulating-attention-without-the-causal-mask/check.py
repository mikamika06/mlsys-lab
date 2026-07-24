import numpy as np


def _oracle(attn_scores, budget):
    scores = np.asarray(attn_scores, dtype=np.float64)
    n = scores.shape[0]
    masked = scores.copy()
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    masked[future] = -np.inf

    shifted = masked - np.max(masked, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    importance = np.sum(probs, axis=0)

    order = sorted(range(n), key=lambda i: (-importance[i], i))
    return np.asarray(order[:budget], dtype=np.int64)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [0.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0, 0.0],
            ]),
            2,
        ),
        (
            np.array([
                [1.0, -2.0, 4.0],
                [3.0, 0.0, 9.0],
                [-1.0, 2.0, 0.0],
            ]),
            1,
        ),
        (
            np.array([
                [0.2, 0.1, 0.3, 0.4, 0.5],
                [0.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            3,
        ),
    ]

    ok = 1.0
    for scores, budget in cases:
        try:
            got = np.asarray(sol.select_heavy_hitters(scores.copy(), budget))
        except Exception:
            ok = 0.0
            break
        if got.dtype != np.int64 or not np.array_equal(got, _oracle(scores, budget)):
            ok = 0.0
            break
    return {"exact_match": ok}
