import numpy as np

def log_softmax(x: np.ndarray) -> np.ndarray:
    """Compute log-softmax along the last axis via the stable x − LSE identity."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    lse = m + np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))
    return x - lse
