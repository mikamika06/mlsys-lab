import numpy as np

def cross_entropy_loss(logits, targets, mask=None):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = np.log(probs + 1e-12)
    idx = targets[..., None]
    log_target = np.take_along_axis(log_probs, idx, axis=-1).squeeze(-1)
    ce = -log_target
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        ce = ce * mask
        denom = np.sum(mask, axis=-1)
        loss = np.where(denom>0, np.sum(ce, axis=-1)/denom, 0.0)
    else:
        loss = np.mean(ce, axis=-1)
    return loss.astype(np.float32)
