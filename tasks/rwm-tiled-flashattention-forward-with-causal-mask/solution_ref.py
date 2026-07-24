import numpy as np


def flash_attention_forward(Q, K, V, block_size=2):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    scale = 1.0 / np.sqrt(float(d))
    O = np.zeros((n, d), dtype=np.float64)

    for qs in range(0, n, block_size):
        qe = min(qs + block_size, n)
        q_block = Q[qs:qe]

        rows = qe - qs
        m = np.full(rows, -np.inf, dtype=np.float64)
        l = np.zeros(rows, dtype=np.float64)
        acc = np.zeros((rows, d), dtype=np.float64)

        for ks in range(0, n, block_size):
            ke = min(ks + block_size, n)
            k_block = K[ks:ke]
            v_block = V[ks:ke]

            scores = q_block @ k_block.T * scale

            row_ids = np.arange(qs, qe)[:, None]
            col_ids = np.arange(ks, ke)[None, :]
            scores = np.where(col_ids > row_ids, -np.inf, scores)

            block_max = np.max(scores, axis=1)
            new_m = np.maximum(m, block_max)

            old_scale = np.exp(m - new_m)
            exp_scores = np.exp(scores - new_m[:, None])

            l = old_scale * l + np.sum(exp_scores, axis=1)
            acc = old_scale[:, None] * acc + exp_scores @ v_block
            m = new_m

        O[qs:qe] = acc / l[:, None]

    return O
