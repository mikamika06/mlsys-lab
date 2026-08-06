import math

def sdpa_single_head(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    seq_len = len(Q)
    d_head = len(Q[0])
    d_v = len(V[0])

    scale = math.sqrt(d_head)

    scores = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            acc = 0.0
            for k in range(d_head):
                acc += Q[i][k] * K[j][k]
            scores[i][j] = acc / scale

    softmax = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
    for i in range(seq_len):
        max_val = float('-inf')
        for j in range(seq_len):
            if scores[i][j] > max_val:
                max_val = scores[i][j]

        sum_exp = 0.0
        for j in range(seq_len):
            e_val = math.exp(scores[i][j] - max_val)
            softmax[i][j] = e_val
            sum_exp += e_val

        for j in range(seq_len):
            softmax[i][j] = softmax[i][j] / sum_exp

    out = [[0.0 for _ in range(d_v)] for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(d_v):
            acc = 0.0
            for k in range(seq_len):
                acc += softmax[i][k] * V[k][j]
            out[i][j] = acc

    return out
