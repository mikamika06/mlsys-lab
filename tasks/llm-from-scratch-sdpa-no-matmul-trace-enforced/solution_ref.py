import numpy as np
from math import exp, sqrt


def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Single-head scaled dot-product attention, from scratch (explicit loops).

    No matmul / @ / dot / einsum: every contraction is an explicit Python loop.
    Softmax is max-shifted for numerical stability.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    S, d = Q.shape
    Sk = K.shape[0]
    dv = V.shape[1]
    scale = 1.0 / sqrt(d)

    # 1) scores[i, j] = (Q_i . K_j) * scale  -- inner product spelled out
    scores = np.empty((S, Sk), dtype=np.float64)
    for i in range(S):
        for j in range(Sk):
            acc = 0.0
            for t in range(d):
                acc += Q[i, t] * K[j, t]
            scores[i, j] = acc * scale

    # 2) stable row softmax  +  3) O[i, t] = sum_j P[i, j] * V[j, t]
    out = np.empty((S, dv), dtype=np.float64)
    for i in range(S):
        m = scores[i, 0]
        for j in range(1, Sk):
            if scores[i, j] > m:
                m = scores[i, j]
        denom = 0.0
        w = np.empty(Sk, dtype=np.float64)
        for j in range(Sk):
            w[j] = exp(scores[i, j] - m)
            denom += w[j]
        for j in range(Sk):
            w[j] /= denom
        for t in range(dv):
            acc = 0.0
            for j in range(Sk):
                acc += w[j] * V[j, t]
            out[i, t] = acc
    return out
