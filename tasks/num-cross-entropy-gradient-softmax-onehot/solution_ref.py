import numpy as np


def cross_entropy_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of the mean softmax cross-entropy w.r.t. ``logits``."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = z.shape[0]

    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    p = e / np.sum(e, axis=-1, keepdims=True)

    p[np.arange(n), y] -= 1.0
    return p / n
