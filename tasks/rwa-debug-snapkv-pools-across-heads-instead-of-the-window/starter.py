import numpy as np


def select_snapkv_indices(attn: np.ndarray, k: int) -> np.ndarray:
    # TODO: incorrect axis. This averages tokens within each head, returning
    # head rankings instead of observation-window token rankings.
    scores = np.mean(np.asarray(attn, dtype=np.float64), axis=1)
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k].astype(np.int64))
