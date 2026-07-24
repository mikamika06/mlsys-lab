import numpy as np


def fused_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-example cross-entropy loss ell_i = logsumexp(logits[i]) - logits[i, targets[i]],
    computed via the numerically-stable log-sum-exp trick (fully vectorised)."""
    raise NotImplementedError('your code here')
