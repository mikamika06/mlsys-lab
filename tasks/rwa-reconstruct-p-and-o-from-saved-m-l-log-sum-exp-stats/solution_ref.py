import math


def flash_forward_reconstruct(Q, K, V, m, l):
    n = len(Q)
    d = len(Q[0])
    k_len = len(K)
    dv = len(V[0])

    S = [[0.0] * k_len for _ in range(n)]
    scale = 1.0 / math.sqrt(d)
    for i in range(n):
        for j in range(k_len):
            dot_val = 0.0
            for k in range(d):
                dot_val += Q[i][k] * K[j][k]
            S[i][j] = dot_val * scale

    P = [[0.0] * k_len for _ in range(n)]
    for i in range(n):
        for j in range(k_len):
            P[i][j] = math.exp(S[i][j] - m[i]) / l[i]

    O = [[0.0] * dv for _ in range(n)]
    for i in range(n):
        for j in range(dv):
            acc = 0.0
            for k in range(k_len):
                acc += P[i][k] * V[k][j]
            O[i][j] = acc

    return P, O
