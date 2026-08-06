import numpy as np


def select_2_4_mask(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W)
    rows, cols = W.shape
    mask = np.zeros((rows, cols), dtype=np.int64)

    groups = W.reshape(rows, cols // 4, 4)
    order = np.argsort(-np.abs(groups), axis=2, kind="stable")
    top = order[:, :, :2]

    group_mask = np.zeros_like(groups, dtype=np.int64)
    np.put_along_axis(group_mask, top, 1, axis=2)

    return group_mask.reshape(rows, cols)
