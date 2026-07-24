import numpy as np


def streaming_attention(Q, K, V, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    m = V.shape[1]
    scale = 1.0 / np.sqrt(d)

    out = np.zeros((n, m), dtype=np.float64)
    running_max = np.full(n, -np.inf, dtype=np.float64)
    running_sum = np.zeros(n, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = (Q @ K[start:end].T) * scale

        block_max = np.max(scores, axis=1)
        new_max = np.maximum(running_max, block_max)

        old_scale = np.exp(running_max - new_max)
        block_exp = np.exp(scores - new_max[:, None])
        new_sum = running_sum * old_scale + np.sum(block_exp, axis=1)

        out = (
            out * (running_sum * old_scale / np.where(new_sum == 0, 1, new_sum))[:, None]
            + (block_exp @ V[start:end]) / np.where(new_sum == 0, 1, new_sum)[:, None]
        )

        running_max = new_max
        running_sum = new_sum

    return out
