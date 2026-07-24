import numpy as np

def log_softmax(x):
    """Compute log(softmax(x)) along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return np.log(probs)
