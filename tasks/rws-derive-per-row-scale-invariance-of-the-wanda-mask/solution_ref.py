import numpy as np


def wanda_mask(W: np.ndarray, col_norms: np.ndarray, keep_ratio: float) -> np.ndarray:
    scores = np.abs(np.asarray(W, dtype=np.float64)) * np.asarray(col_norms, dtype=np.float64)[None, :]
    rows, cols = scores.shape
    k = max(1, int(round(cols * keep_ratio)))

    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        order = np.argsort(-scores[i], kind="stable")
        mask[i, order[:k]] = True
    return mask
