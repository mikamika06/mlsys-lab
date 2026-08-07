import math


def flash_backward(Q, K, V, dO, lse):
    n = len(Q)
    d = len(Q[0])
    dv = len(V[0])
    scale = 1.0 / math.sqrt(d)

    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += Q[i][k] * K[j][k]
            scores[i][j] = dot * scale

    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            P[i][j] = math.exp(scores[i][j] - lse[i])

    dV = [[0.0] * dv for _ in range(n)]
    for a in range(n):
        for b in range(dv):
            acc = 0.0
            for c in range(n):
                acc += P[c][a] * dO[c][b]
            dV[a][b] = acc

    dP = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for c in range(dv):
                acc += dO[i][c] * V[j][c]
            dP[i][j] = acc

    correction = [[0.0] for _ in range(n)]
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += dP[i][j] * P[i][j]
        correction[i][0] = acc

    dS = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dS[i][j] = P[i][j] * (dP[i][j] - correction[i][0])

    dQ = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for k in range(d):
            acc = 0.0
            for j in range(n):
                acc += dS[i][j] * K[j][k]
            dQ[i][k] = acc * scale

    dK = [[0.0] * d for _ in range(n)]
    for j in range(n):
        for k in range(d):
            acc = 0.0
            for i in range(n):
                acc += dS[i][j] * Q[i][k]
            dK[j][k] = acc * scale

    return dQ, dK, dV
