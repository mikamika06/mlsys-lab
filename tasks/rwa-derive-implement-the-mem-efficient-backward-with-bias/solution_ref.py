import math
import numpy as np


def biased_flash_backward(Q, K, V, B, dO, m, l):
    """Memory-efficient attention backward with an additive bias (e.g.
    ALiBi or a mask bias), given only the saved row statistics (m, l)
    from the forward pass -- never a cached probability matrix.

    Q, K, V   : (n, d)
    B         : (n, n) additive bias, added to the scaled scores before
                softmax on the forward pass. Fixed (no gradient wanted).
    dO        : (n, d) upstream gradient w.r.t. the forward output O.
    m, l      : (n,) row max and row softmax-normalizer saved during the
                forward pass, i.e. S = Q@K.T/sqrt(d) + B,
                m = rowmax(S), l = rowsum(exp(S - m)).

    Returns (dQ, dK, dV), each shaped like Q, K, V respectively.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    n, d = Q.shape
    scale = math.sqrt(float(d))

    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            s_val = 0.0
            for k in range(d):
                s_val += Q[i, k] * K[j, k]
            s_val = s_val / scale + B[i, j]
            P[i, j] = math.exp(s_val - m[i]) / l[i]

    dV = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        for k in range(d):
            val = 0.0
            for j in range(n):
                val += P[j, i] * dO[j, k]
            dV[i, k] = val

    dP = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = 0.0
            for k in range(d):
                val += dO[i, k] * V[j, k]
            dP[i, j] = val

    dS = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        corr = 0.0
        for j in range(n):
            corr += dP[i, j] * P[i, j]
        for j in range(n):
            dS[i, j] = P[i, j] * (dP[i, j] - corr)

    dQ = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        for k in range(d):
            val = 0.0
            for j in range(n):
                val += dS[i, j] * K[j, k]
            dQ[i, k] = val / scale

    dK = np.zeros((n, d), dtype=np.float64)
    for j in range(n):
        for k in range(d):
            val = 0.0
            for i in range(n):
                val += dS[i, j] * Q[i, k]
            dK[j, k] = val / scale

    return (
        np.asarray(dQ, dtype=np.float64),
        np.asarray(dK, dtype=np.float64),
        np.asarray(dV, dtype=np.float64),
    )
