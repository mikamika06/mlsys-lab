import math
import numpy as np

def log_softmax(x):
    """Numerically stable log-softmax via the log-sum-exp trick.

    log_softmax(x_i) = x_i - m - log(sum(exp(x_j - m))),  m = max(x).
    """
    x = np.asarray(x, dtype=np.float64)
    m = -float('inf')
    for val in x.flat:
        if val > m:
            m = val
    total = 0.0
    for val in x.flat:
        total += math.exp(val - m)
    offset = m + math.log(total)
    res = np.empty_like(x, dtype=np.float64)
    for i, val in enumerate(x.flat):
        res.flat[i] = val - offset
    return res
