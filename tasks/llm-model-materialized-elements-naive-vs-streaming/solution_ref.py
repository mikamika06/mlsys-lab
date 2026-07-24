import numpy as np

def softmax_streaming(logits: np.ndarray) -> np.ndarray:
    """Stable vectorized softmax applied row‑wise."""
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / s
