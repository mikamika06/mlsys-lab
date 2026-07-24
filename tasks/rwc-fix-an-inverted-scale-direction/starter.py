import numpy as np


def migrate_scale(X: np.ndarray, W: np.ndarray, s: np.ndarray):
    # TODO: this implementation applies the migration in the inverted direction.
    # It pushes large activation channels larger instead of absorbing them into
    # the weights.
    X_new = np.asarray(X, dtype=np.float64) / np.asarray(s, dtype=np.float64)[None, :]
    W_new = np.asarray(W, dtype=np.float64) * np.asarray(s, dtype=np.float64)[:, None]
    return X_new, W_new
