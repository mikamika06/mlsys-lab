import numpy as np

def logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    """Stable log‑sum‑exp over the given axis."""
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))
    else:
        m = np.max(x, axis=axis, keepdims=True)
        sum_exp = np.sum(np.exp(x - m), axis=axis, keepdims=False)
        return (m.squeeze(axis) + np.log(sum_exp))

def log_softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    """Stable log‑softmax over the given axis."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return (x - m) - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
