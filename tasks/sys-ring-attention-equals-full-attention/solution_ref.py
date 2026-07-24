import numpy as np


def ring_attention(Q, K, V, ranks):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n = Q.shape[0]
    d = Q.shape[1]
    dv = V.shape[1]

    blocks_k = np.array_split(K, ranks)
    blocks_v = np.array_split(V, ranks)

    m = np.full(n, -np.inf, dtype=np.float64)
    l = np.zeros(n, dtype=np.float64)
    o = np.zeros((n, dv), dtype=np.float64)

    for step in range(ranks):
        block_index = (np.arange(ranks) + step) % ranks
        for idx in block_index[:1]:
            Kb = blocks_k[int(idx)]
            Vb = blocks_v[int(idx)]

            scores = (Q @ Kb.T) / np.sqrt(d)
            mb = np.max(scores, axis=1)

            new_m = np.maximum(m, mb)
            old_scale = np.exp(m - new_m)
            block_exp = np.exp(scores - new_m[:, None])

            new_l = old_scale * l + np.sum(block_exp, axis=1)
            o = (
                old_scale[:, None] * l[:, None] * o
                + block_exp @ Vb
            ) / new_l[:, None]

            l = new_l
            m = new_m

    return o
