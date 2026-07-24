import numpy as np

def log_softmax(x):
    """Compute log-softmax of x along the last axis."""
    e_x = np.exp(x)
    return np.log(e_x / e_x.sum(axis=-1, keepdims=True))
