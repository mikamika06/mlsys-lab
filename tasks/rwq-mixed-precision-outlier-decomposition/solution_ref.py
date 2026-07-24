import numpy as np


def _quantize_int8(a):
    mx = float(np.max(np.abs(a)))
    scale = mx / 127.0 if mx != 0 else 1.0
    q = np.round(a / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def mixed_precision_matmul(X: np.ndarray, W: np.ndarray, outlier_cols: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    outlier_cols = np.asarray(outlier_cols, dtype=np.int64)

    mask = np.ones(W.shape[1], dtype=bool)
    mask[outlier_cols] = False

    regular_cols = np.flatnonzero(mask)

    x8, sx = _quantize_int8(X)
    w8, sw = _quantize_int8(W[:, regular_cols])

    regular = (x8.astype(np.int32) @ w8.astype(np.int32)).astype(np.float32)
    regular *= sx * sw

    outlier = (X.astype(np.float16) @ W[:, outlier_cols].astype(np.float16)).astype(np.float32)

    result = np.empty((X.shape[0], W.shape[1]), dtype=np.float32)
    result[:, regular_cols] = regular
    result[:, outlier_cols] = outlier
    return result
