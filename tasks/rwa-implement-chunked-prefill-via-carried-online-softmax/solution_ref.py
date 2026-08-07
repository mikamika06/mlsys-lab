import math


def chunked_causal_prefill(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    chunk_sizes: list[int],
) -> list[list[float]]:
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    starts = [0]
    curr = 0
    for size in chunk_sizes:
        curr += size
        starts.append(curr)
    num_chunks = len(chunk_sizes)

    out = [[0.0] * d for _ in range(n)]

    for t in range(num_chunks):
        s, e = starts[t], starts[t + 1]
        cs = e - s

        m = [-float('inf')] * cs
        l = [0.0] * cs
        acc = [[0.0] * d for _ in range(cs)]

        for u in range(t + 1):
            ks, ke = starts[u], starts[u + 1]
            ks_len = ke - ks

            scores = []
            for i in range(cs):
                row_scores = []
                for j in range(ks_len):
                    dot = 0.0
                    for c in range(d):
                        dot += q[s + i][c] * k[ks + j][c]
                    sc = dot * scale
                    if u == t and j > i:
                        sc = -float('inf')
                    row_scores.append(sc)
                scores.append(row_scores)

            blk_max = []
            for i in range(cs):
                mx = -float('inf')
                for j in range(ks_len):
                    if scores[i][j] > mx:
                        mx = scores[i][j]
                blk_max.append(mx)

            m_new = []
            for i in range(cs):
                if m[i] > blk_max[i]:
                    m_new.append(m[i])
                else:
                    m_new.append(blk_max[i])

            correction = []
            for i in range(cs):
                correction.append(math.exp(m[i] - m_new[i]))

            p = []
            for i in range(cs):
                p_row = []
                for j in range(ks_len):
                    p_row.append(math.exp(scores[i][j] - m_new[i]))
                p.append(p_row)

            l_new = []
            for i in range(cs):
                sum_p = 0.0
                for j in range(ks_len):
                    sum_p += p[i][j]
                l_new.append(l[i] * correction[i] + sum_p)
            l = l_new

            acc_new = []
            for i in range(cs):
                acc_row = []
                for c in range(d):
                    pv_sum = 0.0
                    for j in range(ks_len):
                        pv_sum += p[i][j] * v[ks + j][c]
                    val = acc[i][c] * correction[i] + pv_sum
                    acc_row.append(val)
                acc_new.append(acc_row)
            acc = acc_new

            m = m_new

        for i in range(cs):
            inv_l = 1.0 / l[i]
            for c in range(d):
                out[s + i][c] = acc[i][c] * inv_l

    return out
