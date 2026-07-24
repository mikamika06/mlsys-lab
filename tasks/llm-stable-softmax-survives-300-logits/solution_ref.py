import numpy as np

def stable_softmax(logits):
    """Compute softmax along the last axis, numerically stable.

    Subtracts the per-row maximum before exponentiating to avoid overflow.
    """
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)
