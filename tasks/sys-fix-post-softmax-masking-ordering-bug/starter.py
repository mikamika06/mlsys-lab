import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # TODO: This implementation applies the mask after softmax,
    # which leaves the probabilities unnormalised.
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, bool)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_shift = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_shift, axis=-1, keepdims=True)
    probs = exp_shift / sum_exp
    # Zero out masked positions but do not renormalise
    probs[mask] = 0.0
    return probs
