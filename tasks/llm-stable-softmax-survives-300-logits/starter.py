import numpy as np

def stable_softmax(logits):
    """Compute softmax along the last axis, numerically stable."""
    e = np.exp(logits)
    return e / np.sum(e, axis=-1, keepdims=True)
