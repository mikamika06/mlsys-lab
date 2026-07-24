import numpy as np


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _quantize_tensor(x):
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        scale = 1.0
    q = np.round(x / scale).clip(-127, 127).astype(np.int8)
    return q.astype(np.float64) * scale


def quantized_kv_attention(Q, K, V):
    # TODO: this uses a single scale for all tokens instead of one scale per row.
    K_hat = _quantize_tensor(K)
    V_hat = _quantize_tensor(V)
    scores = Q @ K_hat.T / np.sqrt(K.shape[1])
    return _softmax(scores) @ V_hat
