import math
import numpy as np


def flash_forward_reconstruct(Q, K, V, m, l):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    n, d = Q.shape
    k_len, dv = V.shape

    S = np.zeros((n, k_len), dtype=np.float64)
    scale = 1.0 / math.sqrt(d)
    for i in range(n):
        for j in range(k_len):
            dot_val = 0.0
            for k in range(d):
                dot_val += Q[i, k] * K[j, k]
            S[i, j] = dot_val * scale

    P = np.zeros((n, k_len), dtype=np.float64)
    for i in range(n):
        for j in range(k_len):
            P[i, j] = math.exp(S[i, j] - m[i]) / l[i]

    O = np.zeros((n, dv), dtype=np.float64)
    for i in range(n):
        for j in range(dv):
            acc = 0.0
            for k in range(k_len):
                acc += P[i, k] * V[k, j]
            O[i, j] = acc

    return P, O
