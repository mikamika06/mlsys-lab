import numpy as np

def cross_entropy(logits: np.ndarray, target: int) -> float:
    """Stable cross-entropy using the fused log-sum-exp trick.

    Subtracts the running maximum before exponentiating to prevent
    IEEE-754 overflow, then folds the maximum back into the result.

    Parameters
    ----------
    logits : np.ndarray
        1-D array of shape ``(C,)`` of raw class scores.
    target : int
        Ground-truth class index, ``0 <= target < C``.

    Returns
    -------
    float
        Scalar cross-entropy loss.
    """
    m = np.max(logits)
    log_sum_exp = m + np.log(np.sum(np.exp(logits - m)))
    return float(-(logits[target] - log_sum_exp))
