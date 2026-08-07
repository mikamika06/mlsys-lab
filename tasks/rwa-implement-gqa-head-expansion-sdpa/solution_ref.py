import math


def _softmax(x: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    b = len(x)
    h = len(x[0])
    s1 = len(x[0][0])
    s2 = len(x[0][0][0])

    out = [[[[0.0 for _ in range(s2)] for _ in range(s1)] for _ in range(h)] for _ in range(b)]

    for i in range(b):
        for j in range(h):
            for k in range(s1):
                row = x[i][j][k]
                max_val = row[0]
                for val in row:
                    if val > max_val:
                        max_val = val

                exp_sum = 0.0
                for c in range(s2):
                    val = math.exp(row[c] - max_val)
                    out[i][j][k][c] = val
                    exp_sum += val

                for c in range(s2):
                    out[i][j][k][c] /= exp_sum

    return out


def gqa_head_expansion_attention(
    Q: list[list[list[list[float]]]],
    K: list[list[list[list[float]]]],
    V: list[list[list[list[float]]]],
) -> tuple[list[list[list[list[float]]]], float]:
    batch_size = len(Q)
    seq_q = len(Q[0])
    n_q = len(Q[0][0])
    d = len(Q[0][0][0])

    seq_k = len(K[0])
    n_kv = len(K[0][0])
    n_rep = n_q // n_kv

    K_exp = [[[[0.0 for _ in range(d)] for _ in range(n_q)] for _ in range(seq_k)] for _ in range(batch_size)]
    V_exp = [[[[0.0 for _ in range(d)] for _ in range(n_q)] for _ in range(seq_k)] for _ in range(batch_size)]

    for b in range(batch_size):
        for s in range(seq_k):
            for kv in range(n_kv):
                for r in range(n_rep):
                    q_idx = kv * n_rep + r
                    for dim in range(d):
                        K_exp[b][s][q_idx][dim] = K[b][s][kv][dim]
                        V_exp[b][s][q_idx][dim] = V[b][s][kv][dim]

    Qh = [[[[0.0 for _ in range(d)] for _ in range(seq_q)] for _ in range(n_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for sq in range(seq_q):
            for nq in range(n_q):
                for dim in range(d):
                    Qh[b][nq][sq][dim] = Q[b][sq][nq][dim]

    Kh = [[[[0.0 for _ in range(d)] for _ in range(seq_k)] for _ in range(n_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for sk in range(seq_k):
            for nq in range(n_q):
                for dim in range(d):
                    Kh[b][nq][sk][dim] = K_exp[b][sk][nq][dim]

    Vh = [[[[0.0 for _ in range(d)] for _ in range(seq_k)] for _ in range(n_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for sk in range(seq_k):
            for nq in range(n_q):
                for dim in range(d):
                    Vh[b][nq][sk][dim] = V_exp[b][sk][nq][dim]

    sqrt_d = math.sqrt(d)
    scores = [[[[0.0 for _ in range(seq_k)] for _ in range(seq_q)] for _ in range(n_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for nq in range(n_q):
            for sq in range(seq_q):
                for sk in range(seq_k):
                    dot = 0.0
                    for dim in range(d):
                        dot += Qh[b][nq][sq][dim] * Kh[b][nq][sk][dim]
                    scores[b][nq][sq][sk] = dot / sqrt_d

    weights = _softmax(scores)

    attn_out = [[[[0.0 for _ in range(d)] for _ in range(seq_q)] for _ in range(n_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for nq in range(n_q):
            for sq in range(seq_q):
                for dim in range(d):
                    val = 0.0
                    for sk in range(seq_k):
                        val += weights[b][nq][sq][sk] * Vh[b][nq][sk][dim]
                    attn_out[b][nq][sq][dim] = val

    out = [[[[0.0 for _ in range(d)] for _ in range(n_q)] for _ in range(seq_q)] for _ in range(batch_size)]
    for b in range(batch_size):
        for sq in range(seq_q):
            for nq in range(n_q):
                for dim in range(d):
                    out[b][sq][nq][dim] = attn_out[b][nq][sq][dim]

    memory_ratio = n_kv / n_q
    return out, memory_ratio
