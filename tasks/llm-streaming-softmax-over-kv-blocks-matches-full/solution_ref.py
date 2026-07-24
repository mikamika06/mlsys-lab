import numpy as np


def streaming_softmax_attention(Q, K, V, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    out = np.empty((n, V.shape[1]), dtype=np.float64)

    for i in range(n):
        m = -np.inf
        l = 0.0
        o = np.zeros(V.shape[1], dtype=np.float64)

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            scores = (Q[i] @ K[start:end].T) * scale

            block_max = np.max(scores)
            new_m = max(m, block_max)

            old_scale = 0.0 if l == 0.0 else l * np.exp(m - new_m)
            weights = np.exp(scores - new_m)
            new_l = old_scale + np.sum(weights)

            o = (old_scale * o + weights @ V[start:end]) / new_l
            m = new_m
            l = new_l

        out[i] = o

    return out
