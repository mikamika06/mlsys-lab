import math


def attention_with_score_mod(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    cap: float,
) -> list[list[float]]:
    m = len(Q)
    n = len(K)
    d = len(Q[0])
    v_dim = len(V[0])

    sqrt_d = math.sqrt(d)

    scores = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += Q[i][k] * K[j][k]
            val = dot / sqrt_d
            scores[i][j] = cap * math.tanh(val / cap)

    for i in range(m):
        max_val = scores[i][0]
        for j in range(1, n):
            if scores[i][j] > max_val:
                max_val = scores[i][j]
        for j in range(n):
            scores[i][j] -= max_val

    weights = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            ex = math.exp(scores[i][j])
            weights[i][j] = ex
            row_sum += ex
        for j in range(n):
            weights[i][j] /= row_sum

    result = [[0.0] * v_dim for _ in range(m)]
    for i in range(m):
        for j in range(v_dim):
            s = 0.0
            for k in range(n):
                s += weights[i][k] * V[k][j]
            result[i][j] = s

    return result
