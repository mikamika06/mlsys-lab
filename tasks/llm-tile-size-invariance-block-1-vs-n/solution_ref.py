import math
import numpy as np


def streaming_attention(Q, K, V, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    m = V.shape[1]
    scale = 1.0 / math.sqrt(d)

    out = np.zeros((n, m), dtype=np.float64)
    running_max = np.full(n, -math.inf, dtype=np.float64)
    running_sum = np.zeros(n, dtype=np.float64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block_len = end - start

        scores = np.zeros((n, block_len), dtype=np.float64)
        for i in range(n):
            for j in range(block_len):
                dot_val = 0.0
                for k in range(d):
                    dot_val += Q[i, k] * K[start + j, k]
                scores[i, j] = dot_val * scale

        block_max = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row_max = -math.inf
            for j in range(block_len):
                if scores[i, j] > row_max:
                    row_max = scores[i, j]
            block_max[i] = row_max

        new_max = np.zeros(n, dtype=np.float64)
        for i in range(n):
            new_max[i] = running_max[i] if running_max[i] > block_max[i] else block_max[i]

        old_scale = np.zeros(n, dtype=np.float64)
        for i in range(n):
            old_scale[i] = math.exp(running_max[i] - new_max[i])

        block_exp = np.zeros((n, block_len), dtype=np.float64)
        for i in range(n):
            for j in range(block_len):
                block_exp[i, j] = math.exp(scores[i, j] - new_max[i])

        sum_block_exp = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(block_len):
                s += block_exp[i, j]
            sum_block_exp[i] = s

        new_sum = np.zeros(n, dtype=np.float64)
        for i in range(n):
            new_sum[i] = running_sum[i] * old_scale[i] + sum_block_exp[i]

        block_exp_V = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for k in range(block_len):
                    s += block_exp[i, k] * V[start + k, j]
                block_exp_V[i, j] = s

        new_out = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            denom = 1.0 if new_sum[i] == 0.0 else new_sum[i]
            factor1 = (running_sum[i] * old_scale[i]) / denom
            factor2 = 1.0 / denom
            for j in range(m):
                new_out[i, j] = out[i, j] * factor1 + block_exp_V[i, j] * factor2
        out = new_out

        running_max = new_max
        running_sum = new_sum

    return out
