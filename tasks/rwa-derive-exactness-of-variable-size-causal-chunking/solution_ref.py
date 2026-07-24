import numpy as np


def causal_chunk_attention(Q, K, V, chunks):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    out = np.empty((Q.shape[0], V.shape[1]), dtype=np.float64)
    scale = np.sqrt(Q.shape[1])

    start = 0
    for size in chunks:
        end = start + size
        q_chunk = Q[start:end]
        k_visible = K[:end]
        v_visible = V[:end]

        scores = q_chunk @ k_visible.T / scale
        local_rows = np.arange(start, end)[:, None]
        cols = np.arange(end)[None, :]
        scores = np.where(cols > local_rows, -np.inf, scores)

        scores = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        out[start:end] = weights @ v_visible
        start = end

    return out
