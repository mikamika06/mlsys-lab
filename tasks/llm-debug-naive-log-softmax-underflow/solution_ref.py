import numpy as np

def log_softmax(x):
    """Compute log(softmax(x)) along the last axis (numerically stable)."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True)) + x_max
    return x - log_sum_exp
