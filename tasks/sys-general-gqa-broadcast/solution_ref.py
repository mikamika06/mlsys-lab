import math
import numpy as np


def gqa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Grouped-query attention: query heads are split into n_kv contiguous
    groups of size g = n_q // n_kv, each group sharing one KV head.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, n, d = Q.shape
    n_kv = K.shape[0]
    g = n_q // n_kv
    scale = 1.0 / math.sqrt(d)

    O = np.zeros_like(Q)

    for h in range(n_q):
        kv = h // g
        Q_h = Q[h]
        K_kv = K[kv]
        V_kv = V[kv]

        S = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(d):
                    acc += Q_h[i, k] * K_kv[j, k]
                S[i, j] = acc * scale

        max_S = np.zeros((n, 1), dtype=np.float64)
        for i in range(n):
            m = S[i, 0]
            for j in range(1, n):
                if S[i, j] > m:
                    m = S[i, j]
            max_S[i, 0] = m

        P = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                P[i, j] = math.exp(S[i, j] - max_S[i, 0])

        sum_P = np.zeros((n, 1), dtype=np.float64)
        for i in range(n):
            s_acc = 0.0
            for j in range(n):
                s_acc += P[i, j]
            sum_P[i, 0] = s_acc

        for i in range(n):
            for j in range(n):
                P[i, j] /= sum_P[i, 0]

        O_h = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            for k in range(d):
                acc = 0.0
                for j in range(n):
                    acc += P[i, j] * V_kv[j, k]
                O_h[i, k] = acc

        O[h] = O_h

    return O
