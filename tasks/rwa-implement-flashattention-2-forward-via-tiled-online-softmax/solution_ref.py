import math


def flash_attention_forward(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    block_size: int = 32,
) -> list[list[float]]:
    """FlashAttention-2-style forward pass: tiled online softmax.

    Sweeps Q in row blocks and K/V in column blocks, maintaining a running
    max and running normalizer per query row so the result exactly matches
    dense softmax attention without ever materializing an (N, N) matrix.
    """
    n = len(Q)
    if n == 0:
        return []
    d_k = len(Q[0])
    d_v = len(V[0])
    scale = 1.0 / math.sqrt(d_k)

    O = [[0.0] * d_v for _ in range(n)]
    n_blocks = (n + block_size - 1) // block_size

    for i in range(n_blocks):
        q_lo = i * block_size
        q_hi = min(q_lo + block_size, n)
        br = q_hi - q_lo

        m = [float("-inf")] * br
        l = [0.0] * br
        O_block = [[0.0] * d_v for _ in range(br)]

        for j in range(n_blocks):
            kv_lo = j * block_size
            kv_hi = min(kv_lo + block_size, n)
            bc = kv_hi - kv_lo

            S = []
            for r in range(br):
                q_row = Q[q_lo + r]
                s_row = []
                for c in range(bc):
                    k_row = K[kv_lo + c]
                    dot = sum(q_row[d_idx] * k_row[d_idx] for d_idx in range(d_k))
                    s_row.append(dot * scale)
                S.append(s_row)

            row_max = [max(S[r]) for r in range(br)]
            m_new = [max(m[r], row_max[r]) for r in range(br)]

            P = []
            l_cur = []
            rescale = []
            for r in range(br):
                p_row = [math.exp(S[r][c] - m_new[r]) for c in range(bc)]
                P.append(p_row)
                l_cur.append(sum(p_row))
                rescale.append(math.exp(m[r] - m_new[r]))

            for r in range(br):
                r_scale = rescale[r]
                for v in range(d_v):
                    pv_sum = sum(P[r][c] * V[kv_lo + c][v] for c in range(bc))
                    O_block[r][v] = O_block[r][v] * r_scale + pv_sum
                l[r] = l[r] * r_scale + l_cur[r]
                m[r] = m_new[r]

        for r in range(br):
            for v in range(d_v):
                O[q_lo + r][v] = O_block[r][v] / l[r]

    return O
