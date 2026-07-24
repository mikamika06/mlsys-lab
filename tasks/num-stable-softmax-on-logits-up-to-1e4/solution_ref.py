import numpy as np

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    max_vals = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_vals)
    sums = np.sum(exp_shifted, axis=1, keepdims=True)
    return exp_shifted / sums
