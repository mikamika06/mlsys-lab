import numpy as np


def eckart_young_errors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # TODO: incorrectly uses singular values instead of squared singular values
    # and therefore returns the wrong reconstruction error scale.
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X, compute_uv=False)
    dropped = []
    for k in range(len(s) + 1):
        dropped.append(np.sum(s[k:]))
    return np.asarray(dropped), np.asarray(dropped)
===== END
