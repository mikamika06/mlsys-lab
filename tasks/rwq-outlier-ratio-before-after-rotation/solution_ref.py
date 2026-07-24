import numpy as np


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _ratio(X: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    return peak / rms


def outlier_ratio_before_after_rotation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-token peak/rms ratio (over channels) before and after rotating
    the batch with a normalized Sylvester-Hadamard matrix, X_rot = X @ H^T."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)
    Xrot = X @ H.T
    return _ratio(X), _ratio(Xrot)
