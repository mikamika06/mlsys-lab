import math
import numpy as np


def block_sparse_attention(Q, K, V, block_mask, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    blocks = block_mask.shape[0]

    allowed = np.zeros((n, n), dtype=bool)
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi, bj]:
                r0 = bi * block_size
                c0 = bj * block_size
                allowed[
                    r0:min(n, r0 + block_size),
                    c0:min(n, c0 + block_size),
                ] = True

    scores = np.zeros((n, n), dtype=np.float64)
    sqrt_d = math.sqrt(float(d))
    for i in range(n):
        for j in range(n):
            if allowed[i, j]:
                s = 0.0
                for k in range(d):
                    s += Q[i, k] * K[j, k]
                scores[i, j] = s / sqrt_d
            else:
                scores[i, j] = -float("inf")

    for i in range(n):
        max_val = -float("inf")
        for j in range(n):
            if scores[i, j] > max_val:
                max_val = scores[i, j]
        for j in range(n):
            if scores[i, j] != -float("inf"):
                scores[i, j] -= max_val

    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if scores[i, j] == -float("inf"):
                weights[i, j] = 0.0
            else:
                val = math.exp(scores[i, j])
                weights[i, j] = val
                row_sum += val
        if row_sum > 0.0:
            for j in range(n):
                weights[i, j] /= row_sum

    v_cols = V.shape[1]
    output = np.zeros((n, v_cols), dtype=np.float64)
    for i in range(n):
        for col in range(v_cols):
            s = 0.0
            for j in range(n):
                s += weights[i, j] * V[j, col]
            output[i, col] = s

    active = 0
    for bi in range(blocks):
        for bj in range(blocks):
            if block_mask[bi, bj]:
                active += 1

    ratio = float((n * n * d) / (active * block_size * block_size * d))
    return output.astype(np.float64), ratio
