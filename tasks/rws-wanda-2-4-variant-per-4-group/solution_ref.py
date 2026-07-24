import numpy as np


def wanda_2_4_mask(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    scale = np.sqrt(np.mean(X * X, axis=0))
    scores = np.abs(W) * scale

    m, n = W.shape
    mask = np.zeros((m, n), dtype=np.int64)

    for row in range(m):
        for start in range(0, n, 4):
            order = np.argsort(-scores[row, start:start + 4], kind="stable")
            mask[row, start + order[:2]] = 1

    return mask
