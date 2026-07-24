import numpy as np


def residual_distribution(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    The speculative-decoding rejection (residual) distribution: elementwise
    max(p - q, 0), renormalized to sum to 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.maximum(p - q, 0.0)
    return r / r.sum()
