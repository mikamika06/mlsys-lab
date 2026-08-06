import math
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
    m = logits[0]
    for x in logits:
        if x > m:
            m = x

    total = 0.0
    for x in logits:
        total += math.exp(x - m)

    log_sum_exp = m + math.log(total)
    return float(-(logits[target] - log_sum_exp))
