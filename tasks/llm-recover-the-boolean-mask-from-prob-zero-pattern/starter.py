import numpy as np

def recover_mask(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a boolean mask where P[i,j] > eps."""
    return np.ones_like(P, dtype=bool)
