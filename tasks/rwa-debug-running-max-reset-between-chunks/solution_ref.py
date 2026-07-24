import numpy as np


def chunked_attention(q, chunks):
    q = np.asarray(q, dtype=np.float64)
    m = -np.inf
    l = 0.0
    out = np.zeros(chunks[0][1].shape[1], dtype=np.float64)

    for K, V in chunks:
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        scores = K @ q
        chunk_max = np.max(scores)
        new_m = max(m, chunk_max)

        old_scale = 0.0 if m == -np.inf else np.exp(m - new_m)
        weights = np.exp(scores - new_m)

        out = out * (l * old_scale) + weights @ V
        l = l * old_scale + np.sum(weights)
        m = new_m
        out = out / l

    return out
