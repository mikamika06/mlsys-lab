import math
import numpy as np


def apply_rope(x: np.ndarray, pos: int) -> np.ndarray:
    """
    Apply Rotary Position Embedding to a batch of vectors.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n, d) with even d.
    pos : int
        Token position index used to scale the frequency vector.

    Returns
    -------
    np.ndarray
        Rotated array of the same shape and dtype float64.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if d % 2 != 0:
        raise ValueError("Dimension must be even for RoPE.")

    out = np.empty((n, d), dtype=np.float64)
    half_d = d // 2

    for j in range(half_d):
        if half_d == 1:
            omega_j = 0.01
        else:
            omega_j = 0.01 + j * (0.99 - 0.01) / (half_d - 1)
        theta_j = pos * omega_j
        cos_j = math.cos(theta_j)
        sin_j = math.sin(theta_j)

        even_idx = 2 * j
        odd_idx = 2 * j + 1

        for i in range(n):
            even_val = x[i, even_idx]
            odd_val = x[i, odd_idx]
            out[i, even_idx] = even_val * cos_j - odd_val * sin_j
            out[i, odd_idx] = even_val * sin_j + odd_val * cos_j

    return out
