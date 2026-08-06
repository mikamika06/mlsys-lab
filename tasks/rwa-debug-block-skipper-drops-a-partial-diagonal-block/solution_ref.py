import math
import numpy as np


def block_sparse_causal_attention(q, k, v, block_size):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n, d = q.shape
    nb = n // block_size
    scale = 1.0 / math.sqrt(d)

    acc = np.zeros((n, d), dtype=np.float64)
    m = np.full(n, -float("inf"), dtype=np.float64)
    l = np.zeros(n, dtype=np.float64)

    for bi in range(nb):
        qs, qe = bi * block_size, (bi + 1) * block_size

        for bj in range(bi + 1):
            ks, ke = bj * block_size, (bj + 1) * block_size

            scores = np.zeros((block_size, block_size), dtype=np.float64)
            for i in range(block_size):
                for j in range(block_size):
                    dot_val = 0.0
                    for dd in range(d):
                        dot_val += q[qs + i, dd] * k[ks + j, dd]
                    sc = dot_val * scale
                    if bi == bj and j > i:
                        sc = -float("inf")
                    scores[i, j] = sc

            blk_max = np.zeros(block_size, dtype=np.float64)
            for i in range(block_size):
                mx = -float("inf")
                for j in range(block_size):
                    val = scores[i, j]
                    if val > mx:
                        mx = val
                blk_max[i] = mx

            m_prev = np.zeros(block_size, dtype=np.float64)
            for i in range(block_size):
                m_prev[i] = m[qs + i]

            m_new = np.zeros(block_size, dtype=np.float64)
            correction = np.zeros(block_size, dtype=np.float64)
            for i in range(block_size):
                if m_prev[i] > blk_max[i]:
                    m_new[i] = m_prev[i]
                else:
                    m_new[i] = blk_max[i]
                correction[i] = math.exp(m_prev[i] - m_new[i])

            p = np.zeros((block_size, block_size), dtype=np.float64)
            for i in range(block_size):
                for j in range(block_size):
                    p[i, j] = math.exp(scores[i, j] - m_new[i])

            p_sum = np.zeros(block_size, dtype=np.float64)
            for i in range(block_size):
                s_val = 0.0
                for j in range(block_size):
                    s_val += p[i, j]
                p_sum[i] = s_val

            for i in range(block_size):
                l[qs + i] = l[qs + i] * correction[i] + p_sum[i]

            pv = np.zeros((block_size, d), dtype=np.float64)
            for i in range(block_size):
                for dd in range(d):
                    pv_val = 0.0
                    for j in range(block_size):
                        pv_val += p[i, j] * v[ks + j, dd]
                    pv[i, dd] = pv_val

            for i in range(block_size):
                for dd in range(d):
                    acc[qs + i, dd] = acc[qs + i, dd] * correction[i] + pv[i, dd]
                m[qs + i] = m_new[i]

    result = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        l_val = l[i]
        for dd in range(d):
            result[i, dd] = acc[i, dd] / l_val

    return result
