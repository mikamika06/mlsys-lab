import math
import numpy as np


def attention_checkpoint(Q, K, V, G):
    n, d = Q.shape
    m = K.shape[0]
    
    scale = math.sqrt(d)

    scores = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dot = 0.0
            for k_idx in range(d):
                dot += Q[i, k_idx] * K[j, k_idx]
            scores[i, j] = dot / scale

    max_scores = np.zeros((n, 1), dtype=np.float64)
    for i in range(n):
        mx = scores[i, 0]
        for j in range(1, m):
            if scores[i, j] > mx:
                mx = scores[i, j]
        max_scores[i, 0] = mx

    scores_shifted = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            scores_shifted[i, j] = scores[i, j] - max_scores[i, 0]

    exp_scores = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            exp_scores[i, j] = math.exp(scores_shifted[i, j])

    sum_exp = np.zeros((n, 1), dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += exp_scores[i, j]
        sum_exp[i, 0] = s

    P = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            P[i, j] = exp_scores[i, j] / sum_exp[i, 0]

    v_dim = V.shape[1]
    dV = np.zeros((m, v_dim), dtype=np.float64)
    for i in range(m):
        for j in range(v_dim):
            s = 0.0
            for k_idx in range(n):
                s += P[k_idx, i] * G[k_idx, j]
            dV[i, j] = s

    dP = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k_idx in range(v_dim):
                s += G[i, k_idx] * V[j, k_idx]
            dP[i, j] = s

    ds = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        row_sum = 0.0
        for j in range(m):
            row_sum += dP[i, j] * P[i, j]
        for j in range(m):
            ds[i, j] = P[i, j] * (dP[i, j] - row_sum)

    dQ = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            s = 0.0
            for k_idx in range(m):
                s += ds[i, k_idx] * K[k_idx, j]
            dQ[i, j] = s / scale

    dK = np.zeros((m, d), dtype=np.float64)
    for i in range(m):
        for j in range(d):
            s = 0.0
            for k_idx in range(n):
                s += ds[k_idx, i] * Q[k_idx, j]
            dK[i, j] = s / scale

    reported_memory = int(Q.nbytes + K.nbytes + V.nbytes)
    return dQ.astype(np.float64), dK.astype(np.float64), dV.astype(np.float64), reported_memory
