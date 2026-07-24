import numpy as np


def imatrix_from_calibration(X: np.ndarray) -> np.ndarray:
    """Per-input-channel importance: sum over calibration tokens of activation^2.

    X has shape (n_tokens, n_channels); returns a 1-D array of length n_channels.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.sum(X ** 2, axis=0)
