import math
import numpy as np


def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply Layer Normalization to a 2‑D array using unbiased variance.

    Parameters
    ----------
    x : np.ndarray of shape (n, d)
        Input activations.
    eps : float, optional
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    y : np.ndarray of shape (n, d)
        Normalised activations.  The output has dtype float64.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    d = x.shape[1]
    y = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        sum_x = 0.0
        for j in range(d):
            sum_x += x[i, j]
        mean = sum_x / d

        sum_sq_diff = 0.0
        for j in range(d):
            diff = x[i, j] - mean
            sum_sq_diff += diff * diff
        var = sum_sq_diff / (d - 1)

        denom = math.sqrt(var + eps)
        for j in range(d):
            y[i, j] = (x[i, j] - mean) / denom
    return y
