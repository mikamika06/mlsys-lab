import math
import numpy as np


def flash_backward_dq_dk_dv(Q, K, V, dO, logsumexp, causal):
    n, d = Q.shape
    _, m = V.shape
    scale = math.sqrt(d)

    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if causal and j > i:
                scores[i][j] = -float("inf")
            else:
                dot = 0.0
                for k_idx in range(d):
                    dot += Q[i, k_idx] * K[j, k_idx]
                scores[i][j] = dot / scale

    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            val = scores[i][j] - logsumexp[i]
            P[i][j] = math.exp(val)

    dV = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += P[k][i] * dO[k, j]
            dV[i][j] = s

    dP = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += dO[i, k] * V[j, k]
            dP[i][j] = s

    delta = [0.0] * n
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += dP[i][j] * P[i][j]
        delta[i] = row_sum

    dS = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dS[i][j] = P[i][j] * (dP[i][j] - delta[i])

    dQ = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += dS[i][k] * K[k, j]
            dQ[i][j] = s / scale

    dK = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += dS[k][i] * Q[k, j]
            dK[i][j] = s / scale

    return (
        np.array(dQ, dtype=Q.dtype),
        np.array(dK, dtype=Q.dtype),
        np.array(dV, dtype=Q.dtype),
    )
