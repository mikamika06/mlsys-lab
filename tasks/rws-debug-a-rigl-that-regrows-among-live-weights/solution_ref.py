import numpy as np


def rigl_grow(mask, weights, grads, grow_count):
    del weights
    out = np.asarray(mask, dtype=np.int64).copy()
    zero_idx = np.flatnonzero(out == 0)
    count = min(int(grow_count), int(zero_idx.size))
    if count:
        scores = np.abs(np.asarray(grads, dtype=np.float64)[zero_idx])
        chosen = zero_idx[np.argsort(-scores, kind="stable")[:count]]
        out[chosen] = 1
    return out
