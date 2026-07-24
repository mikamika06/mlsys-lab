import numpy as np

def softplus(x: np.ndarray) -> np.ndarray:
    """Naive softplus that overflows on large positive inputs."""
    return np.log1p(np.exp(x))
