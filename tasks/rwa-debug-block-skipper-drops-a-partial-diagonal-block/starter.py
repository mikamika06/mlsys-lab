import numpy as np


def block_sparse_causal_attention(q, k, v, block_size):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n, d = q.shape
    nb = n // block_size
    scale = 1.0 / np.sqrt(d)

    acc = np.zeros((n, d), dtype=np.float64)
    m = np.full(n, -np.inf, dtype=np.float64)
    l = np.zeros(n, dtype=np.float64)

    for bi in range(nb):
        qs, qe = bi * block_size, (bi + 1) * block_size
        q_blk = q[qs:qe]

        # BUG: only visits key blocks strictly before the query block, so
        # the diagonal (partially-masked) tile j == bi is treated as
        # "empty" and skipped, instead of being computed with a mask.
        for bj in range(bi):
            ks, ke = bj * block_size, (bj + 1) * block_size
            k_blk = k[ks:ke]
            v_blk = v[ks:ke]

            scores = (q_blk @ k_blk.T) * scale

            blk_max = np.max(scores, axis=1)
            m_prev = m[qs:qe]
            m_new = np.maximum(m_prev, blk_max)
            correction = np.exp(m_prev - m_new)
            p = np.exp(scores - m_new[:, None])

            l[qs:qe] = l[qs:qe] * correction + np.sum(p, axis=1)
            acc[qs:qe] = acc[qs:qe] * correction[:, None] + p @ v_blk
            m[qs:qe] = m_new

    return acc / l[:, None]
