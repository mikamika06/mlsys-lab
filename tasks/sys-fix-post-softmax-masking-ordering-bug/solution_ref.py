import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, bool)
    # Set masked positions to -inf before softmax
    masked = np.where(mask, -np.inf, logits)
    max_logits = np.max(masked, axis=-1, keepdims=True)
    exp_shift = np.exp(masked - max_logits)
    sum_exp = np.sum(exp_shift, axis=-1, keepdims=True)
    probs = exp_shift / sum_exp
    return probs
