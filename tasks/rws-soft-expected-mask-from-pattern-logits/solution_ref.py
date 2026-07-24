import numpy as np

def soft_expected_mask(logits: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    patterns = np.asarray(patterns, dtype=np.float64)
    # compute softmax over last axis
    max_vals = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - max_vals)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    return probs @ patterns
