import math


def flash_backward(Q, K, V, dO, m, l):
    scale = math.sqrt(float(len(Q[0])))

    N_q = len(Q)
    N_k = len(K)
    d_k = len(Q[0])
    d_v = len(V[0])

    S = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            dot_val = 0.0
            for d in range(d_k):
                dot_val += Q[i][d] * K[j][d]
            S[i][j] = dot_val / scale

    P = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            P[i][j] = math.exp(S[i][j] - m[i]) / l[i]

    dP = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            dot_val = 0.0
            for v_idx in range(d_v):
                dot_val += dO[i][v_idx] * V[j][v_idx]
            dP[i][j] = dot_val

    correction = [[0.0] for _ in range(N_q)]
    for i in range(N_q):
        acc = 0.0
        for j in range(N_k):
            acc += dP[i][j] * P[i][j]
        correction[i][0] = acc

    dS = [[0.0] * N_k for _ in range(N_q)]
    for i in range(N_q):
        for j in range(N_k):
            dS[i][j] = P[i][j] * (dP[i][j] - correction[i][0])

    dQ = [[0.0] * d_k for _ in range(N_q)]
    for i in range(N_q):
        for d in range(d_k):
            acc = 0.0
            for j in range(N_k):
                acc += dS[i][j] * K[j][d]
            dQ[i][d] = acc / scale

    dK = [[0.0] * d_k for _ in range(N_k)]
    for j in range(N_k):
        for d in range(d_k):
            acc = 0.0
            for i in range(N_q):
                acc += dS[i][j] * Q[i][d]
            dK[j][d] = acc / scale

    dV = [[0.0] * d_v for _ in range(N_k)]
    for j in range(N_k):
        for v_idx in range(d_v):
            acc = 0.0
            for i in range(N_q):
                acc += P[i][j] * dO[i][v_idx]
            dV[j][v_idx] = acc

    return dQ, dK, dV
