import numpy as np


def activation_weighted_2_4_mask(W: np.ndarray, X: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    scales = np.linalg.norm(X, axis=1)
    scores = np.abs(W) * scales[None, :]

    mask = np.zeros(W.shape, dtype=np.int64)
    for i in range(W.shape[0]):
        for start in range(0, W.shape[1], 4):
            keep = np.argsort(scores[i, start:start + 4])[-2:]
            mask[i, start + keep] = 1

    error = float(np.sum((W @ X - (W * mask) @ X) ** 2))
    return mask, error
