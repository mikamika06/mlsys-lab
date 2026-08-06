import math
import numpy as np


def per_channel_scales(X: np.ndarray) -> np.ndarray:
    """
    Compute the per‑channel RMS scale for a calibration tensor X.

    Parameters
    ----------
    X : np.ndarray
        2‑D array of shape (N, C).

    Returns
    -------
    scales : np.ndarray
        1‑D float64 array of length C containing the RMS magnitude of each channel.
    """
    X = np.asarray(X, dtype=np.float64)
    N, C = X.shape
    scales = np.zeros(C, dtype=np.float64)
    for j in range(C):
        acc = 0.0
        for i in range(N):
            val = float(X[i, j])
            acc += val * val
        scales[j] = math.sqrt(acc) / math.sqrt(N)
    return scales
