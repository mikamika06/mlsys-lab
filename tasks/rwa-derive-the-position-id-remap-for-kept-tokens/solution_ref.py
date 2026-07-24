import numpy as np


def _rope(x, positions, theta):
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    freq = theta ** (-np.arange(0, d // 2, dtype=np.float64) * 2.0 / d)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * freq[None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    out = x.copy()
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out


def streaming_rope_attention(q, k, v, kept_indices, theta=10000.0):
    q = np.asarray(q, dtype=np.float64)[kept_indices]
    k = np.asarray(k, dtype=np.float64)[kept_indices]
    v = np.asarray(v, dtype=np.float64)[kept_indices]
    positions = np.arange(len(kept_indices), dtype=np.float64)

    qr = _rope(q, positions, theta)
    kr = _rope(k, positions, theta)

    scores = qr @ kr.T / np.sqrt(q.shape[1])
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ v
