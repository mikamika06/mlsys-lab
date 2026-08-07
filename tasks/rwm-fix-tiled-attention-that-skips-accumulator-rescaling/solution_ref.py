import math


def tiled_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int) -> list[list[float]]:
    """
    Non-causal full attention O = softmax(Q K^T) V, computed by streaming
    K/V in blocks of `block_size` rows while maintaining a per-query running
    max `m`, running denominator `l`, and running numerator accumulator `acc`
    (the standard online-softmax / FlashAttention forward recurrence).

    Q : (n_q, d)
    K : (n_k, d)
    V : (n_k, d_v)
    returns O : (n_q, d_v), float64
    """
    n_q = len(Q)
    d = len(Q[0])
    n_k = len(K)
    d_v = len(V[0])

    m = [-math.inf] * n_q
    l = [0.0] * n_q
    acc = [[0.0] * d_v for _ in range(n_q)]

    for start in range(0, n_k, block_size):
        end = min(start + block_size, n_k)
        K_blk = K[start:end]
        V_blk = V[start:end]
        blk_len = end - start

        # Compute scores = Q @ K_blk.T -> (n_q, blk_len)
        scores = []
        for i in range(n_q):
            row_scores = []
            for j in range(blk_len):
                s = sum(Q[i][k] * K_blk[j][k] for k in range(d))
                row_scores.append(s)
            scores.append(row_scores)

        # Compute block_max = max(scores, axis=1) -> (n_q,)
        block_max = []
        for i in range(n_q):
            block_max.append(max(scores[i]))

        # new_m = maximum(m, block_max)
        new_m = [max(m[i], block_max[i]) for i in range(n_q)]

        # alpha = exp(m - new_m)
        alpha = [math.exp(m[i] - new_m[i]) for i in range(n_q)]

        # p = exp(scores - new_m[:, None]) -> (n_q, blk_len)
        p = []
        for i in range(n_q):
            p_row = [math.exp(scores[i][j] - new_m[i]) for j in range(blk_len)]
            p.append(p_row)

        # l = l * alpha + p.sum(axis=1)
        new_l = []
        for i in range(n_q):
            p_sum = sum(p[i])
            new_l.append(l[i] * alpha[i] + p_sum)
        l = new_l

        # acc = acc * alpha[:, None] + p @ V_blk
        new_acc = []
        for i in range(n_q):
            acc_row = []
            for col in range(d_v):
                p_dot_v = sum(p[i][j] * V_blk[j][col] for j in range(blk_len))
                val = acc[i][col] * alpha[i] + p_dot_v
                acc_row.append(val)
            new_acc.append(acc_row)
        acc = new_acc

        m = new_m

    # Return acc / l[:, None]
    out = []
    for i in range(n_q):
        row = [acc[i][col] / l[i] for col in range(d_v)]
        out.append(row)

    return out
