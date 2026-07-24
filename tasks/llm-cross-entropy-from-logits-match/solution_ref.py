import numpy as np

def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    m = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_probs = logits - m - np.log(sum_exp)
    ce = -log_probs[np.arange(len(targets)), targets]
    return float(np.mean(ce))
