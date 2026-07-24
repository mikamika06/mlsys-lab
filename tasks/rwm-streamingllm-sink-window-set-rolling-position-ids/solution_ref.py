import numpy as np


def streaming_attention(tokens, q, k, v, S, W):
    T = len(tokens)
    sink = np.arange(min(S, T), dtype=np.int64)
    window = np.arange(max(0, T - W), T, dtype=np.int64)
    idx = np.unique(np.concatenate([sink, window]))

    start = max(0, T - W)
    pos = np.where(
        idx < S,
        idx,
        S + idx - start,
    ).astype(np.int64)

    kk = k[idx]
    vv = v[idx]
    logits = q @ kk.T / np.sqrt(k.shape[1])
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    out = weights @ vv

    return idx, pos, out
