import math


def gather_attention(
    k_phys: list[list[list[float]]],
    v_phys: list[list[list[float]]],
    block_table: list[int],
    q: list[float]
) -> list[float]:
    k_logical = []
    for b_idx in block_table:
        for row in k_phys[b_idx]:
            k_logical.append(row)

    v_logical = []
    for b_idx in block_table:
        for row in v_phys[b_idx]:
            v_logical.append(row)

    N = len(k_logical)
    D = len(q)

    scale = 1.0 / math.sqrt(D)
    scores = [0.0] * N
    for i in range(N):
        dot = 0.0
        for j in range(D):
            dot += k_logical[i][j] * q[j]
        scores[i] = dot * scale

    max_score = scores[0]
    for i in range(1, N):
        if scores[i] > max_score:
            max_score = scores[i]

    for i in range(N):
        scores[i] -= max_score

    weights = [0.0] * N
    for i in range(N):
        weights[i] = math.exp(scores[i])

    sum_weights = 0.0
    for i in range(N):
        sum_weights += weights[i]

    for i in range(N):
        weights[i] /= sum_weights

    result = [0.0] * D
    for d in range(D):
        acc = 0.0
        for i in range(N):
            acc += weights[i] * v_logical[i][d]
        result[d] = acc

    return result
