import numpy as np

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    max_vals = logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_vals)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    return probs
