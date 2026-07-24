import numpy as np


def _quantize_symmetric(x, bits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def _attention(K, V, q):
    logits = (K @ q) / np.sqrt(K.shape[1])
    logits = logits - np.max(logits)
    p = np.exp(logits)
    p = p / np.sum(p)
    return p @ V


def kv_config_attention_errors(K, V, q):
    base = _attention(np.asarray(K, dtype=np.float64), np.asarray(V, dtype=np.float64), q)
    result = []
    for kb, vb in [(8, 8), (4, 4), (8, 4)]:
        kq = _quantize_symmetric(K, kb)
        vq = _quantize_symmetric(V, vb)
        result.append(float(np.max(np.abs(_attention(kq, vq, q) - base))))
    return np.asarray(result, dtype=np.float64)
