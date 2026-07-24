import numpy as np


def fused_log_softmax_nll(logits: np.ndarray, targets: np.ndarray):
    """Fused stable log-softmax forward + backward: return (loss, dlogits)."""
    raise NotImplementedError('your code here')
