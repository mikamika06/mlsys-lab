import math


def mem_efficient_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    attn_bias: list[list[float]],
    block_size: int = 64,
) -> list[list[float]]:
    n = len(Q)
    d = len(Q[0])
    dv = len(V[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0] * dv for _ in range(n)]

    for qs in range(0, n, block_size):
        qe = min(n, qs + block_size)
        q = Q[qs:qe]

        rows = qe - qs
        m = [-math.inf] * rows
        l = [0.0] * rows
        acc = [[0.0] * dv for _ in range(rows)]

        for ks in range(0, n, block_size):
            ke = min(n, ks + block_size)
            k_block = [K[idx] for idx in range(ks, ke)]
            b_block = [attn_bias[i][ks:ke] for i in range(qs, qe)]

            scores = []
            for r in range(rows):
                row_scores = []
                for c in range(ke - ks):
                    dot = sum(q[r][p] * k_block[c][p] for p in range(d))
                    score = dot * scale + b_block[r][c]
                    row_scores.append(score)
                scores.append(row_scores)

            block_max = [max(row) for row in scores]
            new_m = [max(m[r], block_max[r]) for r in range(rows)]
            old_scale = [math.exp(m[r] - new_m[r]) for r in range(rows)]

            p = []
            for r in range(rows):
                p_row = [math.exp(scores[r][c] - new_m[r]) for c in range(ke - ks)]
                p.append(p_row)

            new_l = [l[r] * old_scale[r] + sum(p[r]) for r in range(rows)]

            v_block = [V[idx] for idx in range(ks, ke)]
            pv = []
            for r in range(rows):
                pv_row = [
                    sum(p[r][c] * v_block[c][v_idx] for c in range(ke - ks))
                    for v_idx in range(dv)
                ]
                pv.append(pv_row)

            new_acc = []
            for r in range(rows):
                nl = new_l[r]
                if nl == 0.0:
                    factor1 = 0.0
                    factor2 = 0.0
                else:
                    factor1 = (l[r] * old_scale[r]) / nl
                    factor2 = 1.0 / nl

                acc_row = [
                    acc[r][v_idx] * factor1 + pv[r][v_idx] * factor2
                    for v_idx in range(dv)
                ]
                new_acc.append(acc_row)
            acc = new_acc

            m = new_m
            l = new_l

        for r in range(rows):
            out[qs + r] = acc[r]

    return out
