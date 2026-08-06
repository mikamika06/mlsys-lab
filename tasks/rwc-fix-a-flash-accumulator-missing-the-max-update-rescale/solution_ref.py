import math
import numpy as np


def flash_attention_accumulate(q, K, V, block_size):
    q = np.asarray(q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    m = float("-inf")
    s = 0.0
    acc = [0.0] * V.shape[1]

    for start in range(0, K.shape[0], block_size):
        end = min(start + block_size, K.shape[0])
        num_rows = end - start

        scores = []
        for i in range(num_rows):
            row = K[start + i]
            dot = 0.0
            for j in range(q.shape[0]):
                dot += row[j] * q[j]
            scores.append(dot)

        block_m = float("-inf")
        for val in scores:
            if val > block_m:
                block_m = val

        if block_m > m:
            if math.isfinite(m):
                scale = math.exp(m - block_m)
                s *= scale
                for j in range(len(acc)):
                    acc[j] *= scale
            m = block_m

        weights = [math.exp(val - m) for val in scores]
        for w in weights:
            s += w

        for i in range(num_rows):
            w = weights[i]
            row_v = V[start + i]
            for j in range(len(acc)):
                acc[j] += w * row_v[j]

    return np.asarray([val / s for val in acc], dtype=np.float64)
