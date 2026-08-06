import math
import numpy as np


def flash_attention_backward(q, k, v, do, m, l):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    do = np.asarray(do, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)

    N_q, D = q.shape
    N_k, _ = k.shape

    scores = np.empty((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            acc = 0.0
            for d in range(D):
                acc += q[i, d] * k[j, d]
            scores[i, j] = acc

    p = np.empty((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            p[i, j] = math.exp(scores[i, j] - m[i, 0]) / l[i, 0]

    dp = np.empty((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            acc = 0.0
            for d in range(D):
                acc += do[i, d] * v[j, d]
            dp[i, j] = acc

    rowsum = np.empty((N_q, 1), dtype=np.float64)
    for i in range(N_q):
        acc = 0.0
        for j in range(N_k):
            acc += dp[i, j] * p[i, j]
        rowsum[i, 0] = acc

    ds = np.empty((N_q, N_k), dtype=np.float64)
    for i in range(N_q):
        for j in range(N_k):
            ds[i, j] = p[i, j] * (dp[i, j] - rowsum[i, 0])

    dq = np.empty((N_q, D), dtype=np.float64)
    for i in range(N_q):
        for d in range(D):
            acc = 0.0
            for j in range(N_k):
                acc += ds[i, j] * k[j, d]
            dq[i, d] = acc

    dk = np.empty((N_k, D), dtype=np.float64)
    for j in range(N_k):
        for d in range(D):
            acc = 0.0
            for i in range(N_q):
                acc += ds[i, j] * q[i, d]
            dk[j, d] = acc

    dv = np.empty((N_k, D), dtype=np.float64)
    for j in range(N_k):
        for d in range(D):
            acc = 0.0
            for i in range(N_q):
                acc += p[i, j] * do[i, d]
            dv[j, d] = acc

    return dq, dk, dv
