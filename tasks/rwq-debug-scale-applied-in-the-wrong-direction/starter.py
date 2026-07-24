import numpy as np


def apply_migration_scale(X: np.ndarray, W: np.ndarray, s: np.ndarray) -> tuple:
    # BUG: multiplies BOTH X and W by s, instead of dividing X by s.
    # This breaks the X' @ W' == X @ W invariant the migration scale relies on.
    X_prime = X * s
    W_prime = W * s[:, None]
    return X_prime, W_prime
