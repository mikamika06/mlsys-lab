import numpy as np


def layerwise_output_mse(W, W_q, X):
    W = np.asarray(W, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    output_error = (W @ X) - (W_q @ X)
    return float(np.mean(output_error ** 2))
