import numpy as np

def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log‑softmax along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    mx = np.max(x, axis=-1, keepdims=True)
    return -mx + np.log(np.sum(np.exp(x - mx), axis=-1, keepdims=True))
