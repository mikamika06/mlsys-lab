import math


def logsumexp_stable(x: list[float]) -> float:
    """
    Numerically stable log-sum-exp: ``log(sum(exp(x)))``.

    Shifts by the maximum entry before exponentiating, so no intermediate
    value overflows even when ``x`` contains entries far outside the range
    where ``exp`` is representable in float64.
    """
    m = -float("inf")
    for i in range(len(x)):
        val = float(x[i])
        if val > m:
            m = val
    s = 0.0
    for i in range(len(x)):
        s += math.exp(float(x[i]) - m)
    return float(m + math.log(s))
