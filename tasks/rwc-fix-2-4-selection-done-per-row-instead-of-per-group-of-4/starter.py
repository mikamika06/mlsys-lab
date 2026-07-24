import numpy as np


def select_2_4_mask(W: np.ndarray) -> np.ndarray:
    # TODO: This incorrectly selects the global top half of each row.
    # It ignores the requirement that every group of four needs exactly two
    # selected values.
    W = np.asarray(W)
    rows, cols = W.shape
    mask = np.zeros((rows, cols), dtype=np.int64)

    for r in range(rows):
        count = cols // 2
        idx = np.argsort(-np.abs(W[r]), kind="stable")[:count]
        mask[r, idx] = 1

    return mask
