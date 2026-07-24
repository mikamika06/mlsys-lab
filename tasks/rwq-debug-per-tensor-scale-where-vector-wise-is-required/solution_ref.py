import numpy as np


def _quantize_rows(A):
    A = np.asarray(A, dtype=np.float64)
    scales = np.max(np.abs(A), axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.rint(A / scales[:, None])
    return np.clip(q, -127, 127).astype(np.int8), scales


def _quantize_cols(A):
    A = np.asarray(A, dtype=np.float64)
    scales = np.max(np.abs(A), axis=0) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.rint(A / scales[None, :])
    return np.clip(q, -127, 127).astype(np.int8), scales


def int8_matmul_per_channel(X, W):
    xq, xs = _quantize_rows(X)
    wq, ws = _quantize_cols(W)
    acc = xq.astype(np.int32) @ wq.astype(np.int32)
    return acc.astype(np.float64) * xs[:, None] * ws[None, :]
