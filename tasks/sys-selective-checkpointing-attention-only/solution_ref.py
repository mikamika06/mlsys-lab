import math


def attention_checkpoint(Q, K, V, G):
    n = len(Q)
    d = len(Q[0])
    m = len(K)

    scale = math.sqrt(d)

    scores = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k_idx in range(d):
                dot += Q[i][k_idx] * K[j][k_idx]
            scores[i][j] = dot / scale

    max_scores = [[0.0] for _ in range(n)]
    for i in range(n):
        mx = scores[i][0]
        for j in range(1, m):
            if scores[i][j] > mx:
                mx = scores[i][j]
        max_scores[i][0] = mx

    scores_shifted = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            scores_shifted[i][j] = scores[i][j] - max_scores[i][0]

    exp_scores = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            exp_scores[i][j] = math.exp(scores_shifted[i][j])

    sum_exp = [[0.0] for _ in range(n)]
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += exp_scores[i][j]
        sum_exp[i][0] = s

    P = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            P[i][j] = exp_scores[i][j] / sum_exp[i][0]

    v_dim = len(V[0])
    dV = [[0.0] * v_dim for _ in range(m)]
    for i in range(m):
        for j in range(v_dim):
            s = 0.0
            for k_idx in range(n):
                s += P[k_idx][i] * G[k_idx][j]
            dV[i][j] = s

    dP = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k_idx in range(v_dim):
                s += G[i][k_idx] * V[j][k_idx]
            dP[i][j] = s

    ds = [[0.0] * m for _ in range(n)]
    for i in range(n):
        row_sum = 0.0
        for j in range(m):
            row_sum += dP[i][j] * P[i][j]
        for j in range(m):
            ds[i][j] = P[i][j] * (dP[i][j] - row_sum)

    dQ = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            s = 0.0
            for k_idx in range(m):
                s += ds[i][k_idx] * K[k_idx][j]
            dQ[i][j] = s / scale

    dK = [[0.0] * d for _ in range(m)]
    for i in range(m):
        for j in range(d):
            s = 0.0
            for k_idx in range(n):
                s += ds[k_idx][i] * Q[k_idx][j]
            dK[i][j] = s / scale

    # Approximate memory as total bytes of floats in Q, K, V
    element_size = 8
    reported_memory = (n * d + m * d + m * d) * element_size
    return dQ, dK, dV, reported_memory
