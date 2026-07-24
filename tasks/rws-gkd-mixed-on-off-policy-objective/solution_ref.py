import numpy as np


def gkd_mixed_loss(student_logits, on_policy_targets, off_policy_targets, lam):
    x = np.asarray(student_logits, dtype=np.float64)
    on = np.asarray(on_policy_targets, dtype=np.int64)
    off = np.asarray(off_policy_targets, dtype=np.int64)

    shifted = x - np.max(x, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    rows = np.arange(x.shape[0])
    on_ce = -log_probs[rows, on]
    off_ce = -log_probs[rows, off]

    return float(np.mean(lam * on_ce + (1.0 - lam) * off_ce))
