import numpy as np


def rigl_grow(mask, weights, grads, grow_count):
    """Grow RigL connections using the selected scores."""
    out = np.asarray(mask, dtype=np.int64).copy()
    count = min(int(grow_count), len(out))
    scores = np.abs(np.asarray(weights, dtype=np.float64))
    chosen = np.argsort(-scores, kind="stable")[:count]
    out[chosen] = 1
    return out
