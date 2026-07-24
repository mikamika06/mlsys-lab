import numpy as np


def classify_mask(mask, L, H, d_ff, target_heads, target_ff):
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (L, H + d_ff):
        return False
    head_counts = np.sum(arr[:, :H], axis=1)
    ff_counts = np.sum(arr[:, H:], axis=1)
    return bool(
        np.all(head_counts == target_heads)
        and np.all(ff_counts == target_ff)
    )
