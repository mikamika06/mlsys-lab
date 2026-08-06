import math
import numpy as np


def flash_attention_backward(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                              O: np.ndarray, L: np.ndarray, dO: np.ndarray,
                              scale: float):
    """
    Recompute-based FlashAttention backward: recompute P from Q, K, L
    (never read a stored full attention matrix), then apply the softmax
    VJP with the D_i = rowsum(dO * O) correction term.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    dO = np.asarray(dO, dtype=np.float64)

    n, d = Q.shape
    n_k, d_k = K.shape

    S = np.zeros((n, n_k), dtype=np.float64)
    for i in range(n):
        for j in range(n_k):
            s_val = 0.0
            for k_idx in range(d):
                s_val += Q[i, k_idx] * K[j, k_idx]
            S[i, j] = s_val * scale

    P = np.zeros((n, n_k), dtype=np.float64)
    for i in range(n):
        for j in range(n_k):
            P[i, j] = math.exp(S[i, j] - L[i])

    dV = np.zeros((n_k, d), dtype=np.float64)
    for i in range(n_k):
        for j in range(d):
            val = 0.0
            for k_idx in range(n):
                val += P[k_idx, i] * dO[k_idx, j]
            dV[i, j] = val

    dP = np.zeros((n, n_k), dtype=np.float64)
    for i in range(n):
        for j in range(n_k):
            val = 0.0
            for k_idx in range(d):
                val += dO[i, k_idx] * V[j, k_idx]
            dP[i, j] = val

    D = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = 0.0
        for k_idx in range(d):
            val += dO[i, k_idx] * O[i, k_idx]
        D[i] = val

    dS = np.zeros((n, n_k), dtype=np.float64)
    for i in range(n):
        for j in range(n_k):
            dS[i, j] = P[i, j] * (dP[i, j] - D[i])

    dQ = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        for j in range(d):
            val = 0.0
            for k_idx in range(n_k):
                val += dS[i, k_idx] * K[k_idx, j]
            dQ[i, j] = val * scale

    dK = np.zeros((n_k, d), dtype=np.float64)
    for i in range(n_k):
        for j in range(d):
            val = 0.0
            for k_idx in range(n):
                val += dS[k_idx, i] * Q[k_idx, j]
            dK[i, j] = val * scale

    return dQ, dK, dV
