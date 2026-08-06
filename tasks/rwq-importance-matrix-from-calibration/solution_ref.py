import numpy as np


def imatrix_from_calibration(X: np.ndarray) -> np.ndarray:
    """Per-input-channel importance: sum over calibration tokens of activation^2.

    X has shape (n_tokens, n_channels); returns a 1-D array of length n_channels.
    """
    X = np.asarray(X, dtype=np.float64)
    n_tokens = X.shape[0]
    n_channels = X.shape[1]
    result = np.zeros(n_channels, dtype=np.float64)
    for j in range(n_channels):
        acc = 0.0
        for i in range(n_tokens):
            val = X[i, j]
            acc += val * val
        result[j] = acc
    return result
