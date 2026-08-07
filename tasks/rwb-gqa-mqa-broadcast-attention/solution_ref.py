import math


def gqa_broadcast_attention(q: list[list[list[float]]], k: list[list[list[float]]], v: list[list[list[float]]]) -> list[list[list[float]]]:
    H_q = len(q)
    n = len(q[0])
    d = len(q[0][0])
    H_kv = len(k)
    n_rep = H_q // H_kv

    k_rep = []
    for h in range(H_q):
        kv_idx = h // n_rep
        k_rep.append(k[kv_idx])

    v_rep = []
    for h in range(H_q):
        kv_idx = h // n_rep
        v_rep.append(v[kv_idx])

    scale = 1.0 / math.sqrt(d)

    scores = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(H_q)]
    for h in range(H_q):
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for l in range(d):
                    acc += q[h][i][l] * k_rep[h][j][l]
                scores[h][i][j] = acc * scale

    for h in range(H_q):
        for i in range(n):
            max_val = scores[h][i][0]
            for j in range(1, n):
                if scores[h][i][j] > max_val:
                    max_val = scores[h][i][j]
            for j in range(n):
                scores[h][i][j] = math.exp(scores[h][i][j] - max_val)

    w = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(H_q)]
    for h in range(H_q):
        for i in range(n):
            s_sum = 0.0
            for j in range(n):
                s_sum += scores[h][i][j]
            for j in range(n):
                w[h][i][j] = scores[h][i][j] / s_sum

    out = [[[0.0 for _ in range(d)] for _ in range(n)] for _ in range(H_q)]
    for h in range(H_q):
        for i in range(n):
            for l in range(d):
                acc = 0.0
                for j in range(n):
                    acc += w[h][i][j] * v_rep[h][j][l]
                out[h][i][l] = acc
    return out
