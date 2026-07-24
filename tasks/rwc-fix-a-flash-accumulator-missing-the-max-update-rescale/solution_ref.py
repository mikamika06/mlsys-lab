import numpy as np


def flash_attention_accumulate(q, K, V, block_size):
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    m = -np.inf
    s = 0.0
    acc = np.zeros(V.shape[1], dtype=np.float64)

    for start in range(0, K.shape[0], block_size):
        end = min(start + block_size, K.shape[0])
        scores = K[start:end] @ q
        block_m = np.max(scores)

        if block_m > m:
            if np.isfinite(m):
                scale = np.exp(m - block_m)
                s *= scale
                acc *= scale
            m = block_m

        weights = np.exp(scores - m)
        s += np.sum(weights)
        acc += weights @ V[start:end]

    return acc / s
