import numpy as np

def log_softmax(x):
    """Numerically stable log-softmax via the log-sum-exp trick.

    log_softmax(x_i) = x_i - m - log(sum(exp(x_j - m))),  m = max(x).
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return x - (m + np.log(np.sum(np.exp(x - m))))
