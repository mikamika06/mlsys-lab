import math
import numpy as np


def logsumexp_stable(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp: ``log(sum(exp(x)))``.

    Shifts by the maximum entry before exponentiating, so no intermediate
    value overflows even when ``x`` contains entries far outside the range
    where ``exp`` is representable in float64.
    """
    x = np.asarray(x, dtype=np.float64)
    m = -float("inf")
    for i in range(x.size):
        val = float(x[i])
        if val > m:
            m = val
    s = 0.0
    for i in range(x.size):
        s += math.exp(float(x[i]) - m)
    return float(m + math.log(s))
