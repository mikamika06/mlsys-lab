import math
import numpy as np


def gather_attention(k_phys, v_phys, block_table, q):
    k_logical = k_phys[block_table].reshape(-1, k_phys.shape[-1]).astype(np.float64)
    v_logical = v_phys[block_table].reshape(-1, v_phys.shape[-1]).astype(np.float64)
    q = q.astype(np.float64)

    N = k_logical.shape[0]
    D = k_logical.shape[1]

    scale = 1.0 / math.sqrt(D)
    scores = [0.0] * N
    for i in range(N):
        dot = 0.0
        for j in range(D):
            dot += k_logical[i, j] * q[j]
        scores[i] = dot * scale

    max_score = scores[0]
    for i in range(1, N):
        if scores[i] > max_score:
            max_score = scores[i]

    for i in range(N):
        scores[i] -= max_score

    weights = [0.0] * N
    for i in range(N):
        weights[i] = math.exp(scores[i])

    sum_weights = 0.0
    for i in range(N):
        sum_weights += weights[i]

    for i in range(N):
        weights[i] /= sum_weights

    result = [0.0] * D
    for d in range(D):
        acc = 0.0
        for i in range(N):
            acc += weights[i] * v_logical[i, d]
        result[d] = acc

    return np.array(result, dtype=np.float64)
