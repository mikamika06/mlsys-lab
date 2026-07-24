import numpy as np


def ce_backward(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # TODO: incorrectly returns softmax probabilities instead of
    # the cross-entropy gradient p - one_hot(label).
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
===== END
