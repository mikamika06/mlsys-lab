import numpy as np


def compute_imatrix(activations):
    """Compute mean square activation values for each layer."""
    res = {}
    for k, act in activations.items():
        arr = np.asarray(act, dtype=np.float32)
        res[k] = np.mean(np.square(arr), axis=0)
    return res
