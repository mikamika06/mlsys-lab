import numpy as np


def per_token_int8_dequant(X):
    X = np.asarray(X, dtype=np.float64)
    scales = np.max(np.abs(X), axis=1, keepdims=True) / 127.0
    q = np.zeros_like(X, dtype=np.int8)
    np.divide(X, scales, out=np.zeros_like(X), where=scales != 0)
    q = np.rint(np.divide(X, scales, out=np.zeros_like(X), where=scales != 0)).astype(np.int8)
    return q.astype(np.float64) * scales
