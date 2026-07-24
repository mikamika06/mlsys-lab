import numpy as np

def orthogonality_error(Q: np.ndarray) -> float:
    """Return the max-abs element of (Q.T @ Q - I)."""
    n = Q.shape[0]
    R = Q.T @ Q - np.eye(n)
    return float(np.max(np.abs(R)))
