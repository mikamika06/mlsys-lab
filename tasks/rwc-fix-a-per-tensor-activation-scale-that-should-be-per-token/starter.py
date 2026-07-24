import numpy as np


def per_token_int8_dequant(X):
    # TODO: this uses one activation scale for the whole tensor.
    # An outlier token reduces precision for all other tokens.
    X = np.asarray(X, dtype=np.float64)
    scale = np.max(np.abs(X)) / 127.0
    if scale == 0:
        return np.zeros_like(X)
    q = np.rint(X / scale).astype(np.int8)
    return q.astype(np.float64) * scale
