import numpy as np


def _quantize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    scales = np.max(np.abs(x), axis=1, keepdims=True) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    q = np.round(x / scales).clip(-127, 127).astype(np.int8)
    return q.astype(np.float64) * scales


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def quantized_kv_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K_hat = _quantize_rows(K)
    V_hat = _quantize_rows(V)
    scores = Q @ K_hat.T / np.sqrt(K_hat.shape[1])
    return (_softmax(scores) @ V_hat).astype(np.float64)
