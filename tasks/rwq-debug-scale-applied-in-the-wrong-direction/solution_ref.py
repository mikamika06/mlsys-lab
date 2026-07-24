import numpy as np


def apply_migration_scale(X: np.ndarray, W: np.ndarray, s: np.ndarray) -> tuple:
    """
    AWQ-style migration scale: shrink activations, grow the matching weight
    rows, so the product X @ W is unchanged.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    X_prime = X / s
    W_prime = W * s[:, None]
    return X_prime, W_prime
