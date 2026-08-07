import math


def enable_gqa_broadcast_attention(
    Q: list[list[list[list[float]]]],
    K: list[list[list[list[float]]]],
    V: list[list[list[list[float]]]],
) -> list[list[list[list[float]]]]:
    batch_size = len(Q)
    n_q = len(Q[0])
    seq_q = len(Q[0][0])
    d = len(Q[0][0][0])

    n_kv = len(K[0])
    seq_k = len(K[0][0])
    d_v = len(V[0][0][0])
    r = n_q // n_kv

    out = [[[[0.0 for _ in range(d_v)] for _ in range(seq_q)] for _ in range(n_q)] for _ in range(batch_size)]
    sqrt_d = math.sqrt(d)

    for b in range(batch_size):
        for h in range(n_q):
            kv_idx = h // r
            for i in range(seq_q):
                scores = []
                for j in range(seq_k):
                    dot = 0.0
                    for k_dim in range(d):
                        dot += Q[b][h][i][k_dim] * K[b][kv_idx][j][k_dim]
                    scores.append(dot / sqrt_d)

                max_score = scores[0]
                for j in range(1, seq_k):
                    if scores[j] > max_score:
                        max_score = scores[j]

                exps = []
                sum_exp = 0.0
                for j in range(seq_k):
                    val = math.exp(scores[j] - max_score)
                    exps.append(val)
                    sum_exp += val

                weights = [val / sum_exp for val in exps]

                for v_dim in range(d_v):
                    acc = 0.0
                    for j in range(seq_k):
                        acc += weights[j] * V[b][kv_idx][j][v_dim]
                    out[b][h][i][v_dim] = acc

    return out
