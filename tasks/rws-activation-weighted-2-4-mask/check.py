import numpy as np


def _oracle_mask(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    scales = np.linalg.norm(X, axis=1)
    scores = np.abs(W) * scales[None, :]
    mask = np.zeros_like(W, dtype=np.int64)
    for i in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            keep = np.argsort(scores[i, start:start + 4])[-2:]
            mask[i, start + keep] = 1
    return mask


def _magnitude_mask(W):
    W = np.asarray(W, dtype=np.float64)
    mask = np.zeros_like(W, dtype=np.int64)
    for i in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            keep = np.argsort(np.abs(W[i, start:start + 4]))[-2:]
            mask[i, start + keep] = 1
    return mask


def _error(W, X, M):
    diff = W @ X - (W * M) @ X
    return float(np.sum(diff * diff))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[3., 1., 5., 2., 4., 8., 1., 3.]]),
            np.array([[1., 0.], [2., 0.], [0., 1.], [0., 3.],
                      [1., 1.], [1., 0.], [0., 2.], [4., 0.]])
        ),
        (
            np.array([[2., -7., 4., 3.], [8., 1., 2., -6.]]),
            np.array([[1., 1., 0.], [0., 2., 0.],
                      [3., 0., 1.], [0., 0., 4.]])
        ),
        (
            np.arange(32, dtype=np.float64).reshape(2, 16) / 5.0,
            np.eye(16, dtype=np.float64)
        ),
    ]

    max_err = 0.0
    weighted_ok = 1.0

    for W, X in cases:
        try:
            mask, err = sol.activation_weighted_2_4_mask(W, X)
        except Exception:
            return {"mse": 1.0, "weighted_not_worse": 0.0}

        ref_mask = _oracle_mask(W, X)
        ref_err = _error(W, X, ref_mask)
        mag_err = _error(W, X, _magnitude_mask(W))

        max_err = max(
            max_err,
            float(np.sum((np.asarray(mask) - ref_mask) ** 2)),
            abs(float(err) - ref_err),
        )

        if float(err) > mag_err + 1e-10:
            weighted_ok = 0.0

    return {
        "mse": max_err,
        "weighted_not_worse": weighted_ok,
    }
