import numpy as np

def softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))
