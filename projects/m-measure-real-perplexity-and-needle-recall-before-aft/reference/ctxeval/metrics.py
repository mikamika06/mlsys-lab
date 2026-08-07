import numpy as np


def compute_perplexity(logits, targets, pad_token_id=None):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if pad_token_id is not None:
        mask = targets != pad_token_id
        logits = logits[mask]
        targets = targets[mask]
    if len(targets) == 0:
        return 0.0
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    target_probs = np.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    target_probs = np.clip(target_probs, 1e-12, 1.0)
    nll = -np.mean(np.log(target_probs))
    return float(np.exp(nll))
