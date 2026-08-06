import math

def log_softmax(x: list[float]) -> list[float]:
    """Numerically stable log-softmax via the log-sum-exp trick.

    log_softmax(x_i) = x_i - m - log(sum(exp(x_j - m))),  m = max(x).
    """
    m = -float('inf')
    for val in x:
        if val > m:
            m = val
    total = 0.0
    for val in x:
        total += math.exp(val - m)
    offset = m + math.log(total)
    res = []
    for val in x:
        res.append(val - offset)
    return res
