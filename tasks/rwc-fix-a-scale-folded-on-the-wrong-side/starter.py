import numpy as np


def fix_awq_scale(W: np.ndarray, X: np.ndarray, s: np.ndarray) -> np.ndarray:
    # TODO: this incorrectly folds the scale into weights without applying
    # the compensating inverse scale to activations.
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return X @ (W * s.reshape(1, -1)).T
