import numpy as np


def wanda_mask(W, col_norms, keep_ratio):
    W = np.asarray(W, dtype=np.float64)
    col_norms = np.asarray(col_norms, dtype=np.float64)
    scores = np.abs(W) * col_norms[None, :]
    rows, cols = scores.shape
    k = max(1, int(round(cols * keep_ratio)))

    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        indices = np.argsort(-scores[i], kind="stable")[:k]
        mask[i, indices] = True
    return mask
