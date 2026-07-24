import numpy as np


def _oracle(W, col_norms, keep_ratio):
    scores = np.abs(np.asarray(W, dtype=np.float64)) * np.asarray(col_norms, dtype=np.float64)[None, :]
    rows, cols = scores.shape
    k = max(1, int(round(cols * keep_ratio)))
    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        order = np.argsort(-scores[i], kind="stable")
        mask[i, order[:k]] = True
    return mask


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -4.0, 2.0, 0.5], [3.0, 1.0, 1.5, 2.0]]),
            np.array([1.0, 0.5, 2.0, 1.0]),
            0.5,
        ),
        (
            np.array([
                [0.2, 4.0, 1.0, 0.1, 2.0],
                [5.0, 0.3, 0.4, 0.2, 0.1],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]),
            np.array([1.0, 2.0, 0.5, 3.0, 1.5]),
            0.4,
        ),
        (
            np.arange(24, dtype=np.float64).reshape(4, 6) - 7.5,
            np.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5]),
            0.5,
        ),
    ]

    for W, col_norms, ratio in cases:
        try:
            got = np.asarray(sol.wanda_mask(W, col_norms, ratio))
        except Exception:
            return {"exact_match": 0.0}
        if got.dtype != np.bool_:
            return {"exact_match": 0.0}
        if not np.array_equal(got, _oracle(W, col_norms, ratio)):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
