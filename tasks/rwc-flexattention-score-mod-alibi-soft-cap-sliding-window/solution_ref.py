import math
import numpy as np


def flex_attention(Q, K, V, score_mod):
    """FlexAttention: apply score_mod before softmax."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape

    sqrt_d = math.sqrt(d)
    scores = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            acc = 0.0
            for k in range(d):
                acc += Q[i, k] * K[j, k]
            scores[i, j] = acc / sqrt_d

    qi = np.arange(N).reshape(N, 1)
    ki = np.arange(N).reshape(1, N)
    scores = score_mod(scores, qi, ki)

    scores = np.asarray(scores, dtype=np.float64)
    for i in range(N):
        max_val = scores[i, 0]
        for j in range(1, N):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        for j in range(N):
            scores[i, j] -= max_val

    weights = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            weights[i, j] = math.exp(scores[i, j])

    row_sum = np.zeros((N, 1), dtype=np.float64)
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += weights[i, j]
        row_sum[i, 0] = s

    for i in range(N):
        if row_sum[i, 0] == 0.0:
            row_sum[i, 0] = 1.0

    for i in range(N):
        r = row_sum[i, 0]
        for j in range(N):
            weights[i, j] /= r

    res = np.zeros((N, d), dtype=np.float64)
    for i in range(N):
        for j in range(d):
            acc = 0.0
            for k in range(N):
                acc += weights[i, k] * V[k, j]
            res[i, j] = acc

    return res.astype(np.float32)
