import numpy as np


def _rope(x, positions):
    x = np.asarray(x, dtype=np.float64).copy()
    d = x.shape[1]
    half = d // 2
    freqs = 1.0 / (10000.0 ** (2 * np.arange(half) / d))
    angles = positions[:, None] * freqs[None, :]
    c = np.cos(angles)
    s = np.sin(angles)
    even = x[:, 0::2].copy()
    odd = x[:, 1::2].copy()
    x[:, 0::2] = even * c - odd * s
    x[:, 1::2] = even * s + odd * c
    return x


def _attention(q, k, v):
    if q.shape[0] == 0:
        return np.empty_like(q)
    logits = q @ k.T / np.sqrt(q.shape[1])
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs @ v


def packed_rope_attention(q, k, v, cu_seqlens):
    out = np.empty_like(q, dtype=np.float64)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        if end == start:
            continue
        positions = np.arange(end - start, dtype=np.float64)
        rq = _rope(q[start:end], positions)
        rk = _rope(k[start:end], positions)
        out[start:end] = _attention(rq, rk, v[start:end])
    return out
