import numpy as np


def mem_efficient_attention(Q, K, V, attn_bias, block_size=64):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    B = np.asarray(attn_bias, dtype=np.float64)

    n, d = Q.shape
    dv = V.shape[1]
    scale = 1.0 / np.sqrt(d)
    out = np.zeros((n, dv), dtype=np.float64)

    for qs in range(0, n, block_size):
        qe = min(n, qs + block_size)
        q = Q[qs:qe]

        rows = qe - qs
        m = np.full(rows, -np.inf, dtype=np.float64)
        l = np.zeros(rows, dtype=np.float64)
        acc = np.zeros((rows, dv), dtype=np.float64)

        for ks in range(0, n, block_size):
            ke = min(n, ks + block_size)

            scores = q @ K[ks:ke].T * scale + B[qs:qe, ks:ke]
            block_max = np.max(scores, axis=1)

            new_m = np.maximum(m, block_max)
            old_scale = np.exp(m - new_m)
            p = np.exp(scores - new_m[:, None])

            new_l = l * old_scale + np.sum(p, axis=1)
            acc = (acc * (l * old_scale / new_l)[:, None] +
                   (p @ V[ks:ke]) / new_l[:, None])

            m = new_m
            l = new_l

        out[qs:qe] = acc

    return out
