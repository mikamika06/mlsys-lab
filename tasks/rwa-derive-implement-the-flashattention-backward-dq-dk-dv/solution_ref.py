import math
import numpy as np


def flash_backward(Q, K, V, dO, m, l):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    scale = math.sqrt(float(Q.shape[1]))
    
    N_q = Q.shape[0]
    N_k = K.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]

    S = np.zeros((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            dot_val = 0.0
            for d in range(d_k):
                dot_val += Q[i, d] * K[j, d]
            S[i, j] = dot_val / scale

    P = np.zeros((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            P[i, j] = math.exp(S[i, j] - m[i]) / l[i]

    dP = np.zeros((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            dot_val = 0.0
            for v_idx in range(d_v):
                dot_val += dO[i, v_idx] * V[j, v_idx]
            dP[i, j] = dot_val

    correction = np.zeros((N_q, 1), dtype=np.float64)
    for i in range(N_q):
        acc = 0.0
        for j in range(N_k):
            acc += dP[i, j] * P[i, j]
        correction[i, 0] = acc

    dS = np.zeros((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            dS[i, j] = P[i, j] * (dP[i, j] - correction[i, 0])

    dQ = np.zeros((N_q, d_k), dtype=np.float64)
    for i in range(N_q):
        for d in range(d_k):
            acc = 0.0
            for j in range(N_k):
                acc += dS[i, j] * K[j, d]
            dQ[i, d] = acc / scale

    dK = np.zeros((N_k, d_k), dtype=np.float64)
    for j in range(N_k):
        for d in range(d_k):
            acc = 0.0
            for i in range(N_q):
                acc += dS[i, j] * Q[i, d]
            dK[j, d] = acc / scale

    dV = np.zeros((N_k, d_v), dtype=np.float64)
    for j in range(N_k):
        for v_idx in range(d_v):
            acc = 0.0
            for i in range(N_q):
                acc += P[i, j] * dO[i, v_idx]
            dV[j, v_idx] = acc

    return (
        np.asarray(dQ, dtype=np.float64),
        np.asarray(dK, dtype=np.float64),
        np.asarray(dV, dtype=np.float64),
    )
