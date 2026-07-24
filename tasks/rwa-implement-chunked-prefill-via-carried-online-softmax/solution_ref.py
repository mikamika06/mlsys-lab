import numpy as np


def chunked_causal_prefill(q, k, v, chunk_sizes):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n, d = q.shape
    scale = 1.0 / np.sqrt(d)

    starts = np.cumsum([0] + list(chunk_sizes))
    num_chunks = len(chunk_sizes)
    out = np.zeros((n, d), dtype=np.float64)

    for t in range(num_chunks):
        s, e = int(starts[t]), int(starts[t + 1])
        q_chunk = q[s:e]
        cs = e - s

        m = np.full(cs, -np.inf, dtype=np.float64)
        l = np.zeros(cs, dtype=np.float64)
        acc = np.zeros((cs, d), dtype=np.float64)

        rows = np.arange(cs)[:, None]
        cols = np.arange(cs)[None, :]
        diag_mask = cols > rows

        for u in range(t + 1):
            ks, ke = int(starts[u]), int(starts[u + 1])
            k_blk = k[ks:ke]
            v_blk = v[ks:ke]

            scores = (q_chunk @ k_blk.T) * scale
            if u == t:
                scores = np.where(diag_mask, -np.inf, scores)

            blk_max = np.max(scores, axis=1)
            m_new = np.maximum(m, blk_max)
            correction = np.exp(m - m_new)
            p = np.exp(scores - m_new[:, None])

            l = l * correction + np.sum(p, axis=1)
            acc = acc * correction[:, None] + p @ v_blk
            m = m_new

        out[s:e] = acc / l[:, None]

    return out
