import math
import numpy as np


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    scale = math.sqrt(n)
    h_out = np.empty((h.shape[0], h.shape[1]), dtype=np.float64)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h_out[i, j] = h[i, j] / scale
    return h_out


def _ratio(X: np.ndarray) -> np.ndarray:
    rows = X.shape[0]
    cols = X.shape[1]
    res = np.empty((rows,), dtype=np.float64)
    for i in range(rows):
        sum_sq = 0.0
        max_abs = 0.0
        for j in range(cols):
            val = X[i, j]
            sum_sq += val * val
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
        rms = math.sqrt(sum_sq / cols)
        res[i] = max_abs / rms
    return res


def outlier_ratio_before_after_rotation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-token peak/rms ratio (over channels) before and after rotating
    the batch with a normalized Sylvester-Hadamard matrix, X_rot = X @ H^T."""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)
    
    rows = X.shape[0]
    H_rows = H.shape[0]
    H_cols = H.shape[1]
    
    Xrot = np.empty((rows, H_rows), dtype=np.float64)
    for i in range(rows):
        for j in range(H_rows):
            acc = 0.0
            for k in range(d):
                acc += X[i, k] * H[j, k]
            Xrot[i, j] = acc
            
    return _ratio(X), _ratio(Xrot)
