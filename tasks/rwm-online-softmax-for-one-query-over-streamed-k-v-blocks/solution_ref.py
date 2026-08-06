import math
import numpy as np


def online_attention(q, K_blocks, V_blocks):
    q = np.asarray(q, dtype=np.float64)

    m = -float('inf')
    l = 0.0
    acc = None

    dim = q.shape[0]
    scale = math.sqrt(float(dim))

    for K, V in zip(K_blocks, V_blocks):
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)

        num_rows = K.shape[0]
        scores = [0.0] * num_rows
        for i in range(num_rows):
            dot = 0.0
            for j in range(dim):
                dot += K[i, j] * q[j]
            scores[i] = dot / scale

        block_max = -float('inf')
        for s in scores:
            if s > block_max:
                block_max = s

        if m == -float('inf'):
            new_m = block_max
        else:
            new_m = m if m > block_max else block_max

        old_scale = math.exp(m - new_m) if math.isfinite(m) else 0.0

        weights = [0.0] * num_rows
        sum_weights = 0.0
        for i in range(num_rows):
            w = math.exp(scores[i] - new_m)
            weights[i] = w
            sum_weights += w

        new_l = old_scale * l + sum_weights

        v_cols = V.shape[1]
        if acc is None:
            acc = np.zeros(v_cols, dtype=np.float64)

        weighted_v = [0.0] * v_cols
        for c in range(v_cols):
            col_sum = 0.0
            for i in range(num_rows):
                col_sum += weights[i] * V[i, c]
            weighted_v[c] = col_sum

        if new_l == 0.0:
            acc = np.zeros(v_cols, dtype=np.float64)
        else:
            term1 = old_scale * l
            for c in range(v_cols):
                acc[c] = (term1 * acc[c] + weighted_v[c]) / new_l

        l = new_l
        m = new_m

    return acc
