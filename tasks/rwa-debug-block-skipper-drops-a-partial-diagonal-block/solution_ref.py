import math


def block_sparse_causal_attention(q, k, v, block_size):
    n = len(q)
    d = len(q[0])
    nb = n // block_size
    scale = 1.0 / math.sqrt(d)

    acc = [[0.0] * d for _ in range(n)]
    m = [-float('inf')] * n
    l = [0.0] * n

    for bi in range(nb):
        qs, qe = bi * block_size, (bi + 1) * block_size
        q_blk = q[qs:qe]

        # j > bi is fully masked -> skip. j < bi is fully visible -> dense.
        # j == bi is the partial diagonal tile -> compute WITH the mask,
        # never skip it.
        for bj in range(bi + 1):
            ks, ke = bj * block_size, (bj + 1) * block_size
            k_blk = k[ks:ke]
            v_blk = v[ks:ke]

            block_h = len(q_blk)
            block_w = len(k_blk)
            scores = [[0.0] * block_w for _ in range(block_h)]
            for r in range(block_h):
                for c in range(block_w):
                    dot = sum(q_blk[r][p] * k_blk[c][p] for p in range(d))
                    scores[r][c] = dot * scale

            if bj == bi:
                for r in range(block_h):
                    for c in range(block_w):
                        if c > r:
                            scores[r][c] = -float('inf')

            blk_max = [max(row) for row in scores]
            m_prev = m[qs:qe]
            m_new = [max(m_prev[r], blk_max[r]) for r in range(block_h)]
            correction = [math.exp(m_prev[r] - m_new[r]) for r in range(block_h)]
            p = [[math.exp(scores[r][c] - m_new[r]) for c in range(block_w)] for r in range(block_h)]

            for r in range(block_h):
                p_sum = sum(p[r])
                l[qs + r] = l[qs + r] * correction[r] + p_sum

            p_v = [[0.0] * d for _ in range(block_h)]
            for r in range(block_h):
                for col_idx in range(d):
                    s_pv = sum(p[r][c] * v_blk[c][col_idx] for c in range(block_w))
                    p_v[r][col_idx] = s_pv

            for r in range(block_h):
                for col_idx in range(d):
                    acc[qs + r][col_idx] = acc[qs + r][col_idx] * correction[r] + p_v[r][col_idx]

            m[qs:qe] = m_new

    result = [[0.0] * d for _ in range(n)]
    for r in range(n):
        inv_l = 1.0 / l[r]
        for c in range(d):
            result[r][c] = acc[r][c] * inv_l

    return result
