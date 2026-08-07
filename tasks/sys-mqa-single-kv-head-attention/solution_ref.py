import math

def mha_single_kv_head(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    """Compute scaled dot-product attention with a single shared KV head.

    Q : (B, H, S, D)  – queries, H heads
    K : (B, 1, S, D)  – keys, one head
    V : (B, 1, S, D)  – values, one head

    Returns (B, H, S, D) using plain Python lists.
    """
    B = len(Q)
    H = len(Q[0])
    S_q = len(Q[0][0])
    D = len(Q[0][0][0])
    S_k = len(K[0][0])

    scale = D ** -0.5

    output = [[[[0.0 for _ in range(D)] for _ in range(S_q)] for _ in range(H)] for _ in range(B)]

    for b in range(B):
        for h in range(H):
            for i in range(S_q):
                scores_row = [0.0] * S_k
                for j in range(S_k):
                    dot_val = 0.0
                    for d in range(D):
                        dot_val += Q[b][h][i][d] * K[b][0][j][d]
                    scores_row[j] = dot_val * scale

                max_val = scores_row[0]
                for j in range(1, S_k):
                    if scores_row[j] > max_val:
                        max_val = scores_row[j]

                exp_row = [0.0] * S_k
                sum_exp = 0.0
                for j in range(S_k):
                    e = math.exp(scores_row[j] - max_val)
                    exp_row[j] = e
                    sum_exp += e

                weights_row = [0.0] * S_k
                for j in range(S_k):
                    weights_row[j] = exp_row[j] / sum_exp

                for d in range(D):
                    out_val = 0.0
                    for j in range(S_k):
                        out_val += weights_row[j] * V[b][0][j][d]
                    output[b][h][i][d] = out_val

    return output
