import math
import numpy as np


def attention_with_score_mod(Q, K, V, cap):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    m = Q.shape[0]
    n = K.shape[0]
    d = Q.shape[1]
    v_dim = V.shape[1]

    sqrt_d = math.sqrt(d)

    scores = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            dot = 0.0
            for k in range(d):
                dot += Q[i, k] * K[j, k]
            val = dot / sqrt_d
            scores[i, j] = cap * math.tanh(val / cap)

    for i in range(m):
        max_val = scores[i, 0]
        for j in range(1, n):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        for j in range(n):
            scores[i, j] -= max_val

    weights = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        row_sum = 0.0
        for j in range(n):
            ex = math.exp(scores[i, j])
            weights[i, j] = ex
            row_sum += ex
        for j in range(n):
            weights[i, j] /= row_sum

    result = np.zeros((m, v_dim), dtype=np.float64)
    for i in range(m):
        for j in range(v_dim):
            s = 0.0
            for k in range(n):
                s += weights[i, k] * V[k, j]
            result[i, j] = s

    return result
