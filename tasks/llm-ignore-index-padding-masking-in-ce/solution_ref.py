import numpy as np

def masked_cross_entropy(logits: np.ndarray,
                         targets: np.ndarray,
                         ignore_index: int = -100) -> float:
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    # stable log‑softmax
    max_logits = logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    log_probs = logits - max_logits - np.log(sum_exp)
    neg_log_likelihood = -log_probs[np.arange(len(targets)), targets]
    mask = targets != ignore_index
    if not mask.any():
        return 0.0
    return float(neg_log_likelihood[mask].mean())
