import numpy as np


def stable_softmax_argmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    probs = e / np.sum(e, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
