import math

def cross_entropy(logits: list[float], target: int) -> float:
    """Stable cross-entropy using the fused log-sum-exp trick.

    Subtracts the running maximum before exponentiating to prevent
    IEEE-754 overflow, then folds the maximum back into the result.

    Parameters
    ----------
    logits : list[float]
        List of raw class scores.
    target : int
        Ground-truth class index, ``0 <= target < len(logits)``.

    Returns
    -------
    float
        Scalar cross-entropy loss.
    """
    m = max(logits)
    sum_exp = 0.0
    for x in logits:
        sum_exp += math.exp(x - m)
    log_sum_exp = m + math.log(sum_exp)
    return float(-(logits[target] - log_sum_exp))
