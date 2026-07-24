import numpy as np


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    The distribution to resample from when a speculative-decoding draft
    token is rejected.

    BUG: this just returns the target distribution p unchanged, instead of
    the normalized residual max(p - q, 0). Fix it.
    """
    p = np.asarray(p, dtype=np.float64)
    return p.copy()
