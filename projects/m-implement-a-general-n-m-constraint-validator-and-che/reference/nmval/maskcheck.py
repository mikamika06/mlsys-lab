import numpy as np


def check_real_mask(mask, n=2, m=4):
    arr = np.asarray(mask, dtype=bool)
    flat = arr.reshape(-1, m)
    counts = np.sum(flat, axis=1)
    valid = np.all(counts <= n)
    return bool(valid), counts.tolist()
