import numpy as np

def cross_entropy_loss(logits, targets, mask=None):
    logits = np.asarray(logits, dtype=np.float64)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = np.log(probs + 1e-12)
    idx = np.asarray(targets, dtype=np.int64)[..., None]
    log_target = np.take_along_axis(log_probs, idx, axis=-1).squeeze(-1)
    ce = -log_target
    if mask is not None:
        ce = ce * mask.astype(bool)
    # Bug: sum instead of mean
    loss = np.sum(ce, axis=-1)
    return loss.astype(np.float32)
