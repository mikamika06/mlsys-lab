import numpy as np


def fused_log_softmax_nll(logits: np.ndarray, targets: np.ndarray):
    """Fused stable log-softmax forward + backward: mean NLL loss and its gradient."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n = logits.shape[0]

    m = np.max(logits, axis=1, keepdims=True)
    shifted = logits - m
    lse = m[:, 0] + np.log(np.sum(np.exp(shifted), axis=1))
    log_probs = logits - lse[:, None]

    idx = np.arange(n)
    loss = -float(np.mean(log_probs[idx, targets]))

    probs = np.exp(log_probs)
    dlogits = probs.copy()
    dlogits[idx, targets] -= 1.0
    dlogits /= n

    return loss, dlogits
