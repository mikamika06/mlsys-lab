import numpy as np


def select_snapkv_indices(attn: np.ndarray, k: int) -> np.ndarray:
    scores = np.mean(np.asarray(attn, dtype=np.float64), axis=0)
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k].astype(np.int64))
