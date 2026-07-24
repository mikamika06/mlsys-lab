import numpy as np


def logsumexp_stable(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp: ``log(sum(exp(x)))``.

    Shifts by the maximum entry before exponentiating, so no intermediate
    value overflows even when ``x`` contains entries far outside the range
    where ``exp`` is representable in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))
