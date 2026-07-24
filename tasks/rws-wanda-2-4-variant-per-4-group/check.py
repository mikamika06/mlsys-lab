import numpy as np


def _oracle(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    scale = np.sqrt(np.mean(X * X, axis=0))
    scores = np.abs(W) * scale
    mask = np.zeros_like(W, dtype=np.int64)
    groups = W.shape[1] // 4
    for row in range(W.shape[0]):
        for g in range(groups):
            start = g * 4
            vals = scores[row, start:start + 4]
            order = sorted(range(4), key=lambda i: (-vals[i], i))
            for idx in order[:2]:
                mask[row, start + idx] = 1
    return mask


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, -4.0, 2.0, 3.0],
                      [5.0, 1.0, -2.0, 4.0]]),
            np.array([[1.0, 2.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0]])
        ),
        (
            np.array([[0.5, -1.5, 3.0, 2.0, 8.0, 1.0, -4.0, 2.0],
                      [-2.0, 6.0, 1.0, 0.0, 1.0, 3.0, 5.0, -7.0]]),
            np.array([
                [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 4.0],
                [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0],
            ])
        ),
        (
            np.arange(24, dtype=np.float64).reshape(3, 8) - 5.0,
            np.ones((4, 8), dtype=np.float64)
        ),
    ]

    exact = 1.0
    valid = 1.0

    for W, X in cases:
        try:
            got = np.asarray(sol.wanda_2_4_mask(W, X))
        except Exception:
            return {"exact_match": 0.0, "valid_2_4": 0.0}

        ref = _oracle(W, X)
        if got.shape != ref.shape or not np.array_equal(got, ref):
            exact = 0.0

        if got.shape != ref.shape:
            valid = 0.0
            continue
        if not np.all((got == 0) | (got == 1)):
            valid = 0.0
        for row in range(got.shape[0]):
            for start in range(0, got.shape[1], 4):
                if int(np.sum(got[row, start:start + 4])) != 2:
                    valid = 0.0

    return {"exact_match": exact, "valid_2_4": valid}
