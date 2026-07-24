import numpy as np


def fused_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-example cross-entropy loss ell_i = logsumexp(logits[i]) - logits[i, targets[i]],
    computed via the numerically-stable log-sum-exp trick (fully vectorised)."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    m = np.max(logits, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.sum(np.exp(logits - m), axis=1))
    tgt_logit = logits[np.arange(logits.shape[0]), targets]
    return lse - tgt_logit
