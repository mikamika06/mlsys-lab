import numpy as np


def outlier_ratio_before_after_rotation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-token peak/rms ratio (over channels) before and after rotating
    the batch with a normalized Sylvester-Hadamard matrix, X_rot = X @ H^T.

    Returns (ratio_before, ratio_after), each a 1-D float64 array of length
    X.shape[0].
    """
    raise NotImplementedError('your code here')
