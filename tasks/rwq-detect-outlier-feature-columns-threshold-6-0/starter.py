import numpy as np


def detect_outlier_columns(X: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    """
    Return the sorted, unique column indices j of X where max_i |X[i, j]| >= threshold.
    """
    raise NotImplementedError('your code here')
