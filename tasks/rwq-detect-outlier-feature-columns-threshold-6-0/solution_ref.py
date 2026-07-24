import numpy as np


def detect_outlier_columns(X: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """
    Return the sorted, unique column indices j where max_i |X[i, j]| >= threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    absmax = np.max(np.abs(X), axis=0)
    idx = np.nonzero(absmax >= threshold)[0]
    return np.sort(idx).astype(np.int64)
