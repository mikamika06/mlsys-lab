import numpy as np


def online_attention(q, K_blocks, V_blocks):
    q = np.asarray(q, dtype=np.float64)

    m = -np.inf
    l = 0.0
    acc = None

    scale = np.sqrt(float(q.shape[0]))

    for K, V in zip(K_blocks, V_blocks):
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)

        scores = K @ q / scale
        block_max = float(np.max(scores))
        new_m = max(m, block_max)

        old_scale = np.exp(m - new_m) if np.isfinite(m) else 0.0
        weights = np.exp(scores - new_m)

        new_l = old_scale * l + float(np.sum(weights))

        if acc is None:
            acc = np.zeros(V.shape[1], dtype=np.float64)

        acc = (old_scale * l * acc + weights @ V) / new_l
        l = new_l
        m = new_m

    return acc
