import math
import numpy as np


def scaled_dot_product_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    batch_q = q.shape[0]
    d = q.shape[1]
    batch_k = k.shape[0]
    v_cols = v.shape[1]

    sqrt_d = math.sqrt(d)

    logits = np.zeros((batch_q, batch_k), dtype=np.float64)
    for i in range(batch_q):
        for j in range(batch_k):
            dot_val = 0.0
            for l in range(d):
                dot_val += q[i, l] * k[j, l]
            logits[i, j] = dot_val / sqrt_d

    for i in range(batch_q):
        max_val = logits[i, 0]
        for j in range(1, batch_k):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        for j in range(batch_k):
            logits[i, j] = math.exp(logits[i, j] - max_val)

    weights = np.zeros((batch_q, batch_k), dtype=np.float64)
    for i in range(batch_q):
        sum_val = 0.0
        for j in range(batch_k):
            sum_val += logits[i, j]
        for j in range(batch_k):
            weights[i, j] = logits[i, j] / sum_val

    result = np.zeros((batch_q, v_cols), dtype=np.float64)
    v_rows = v.shape[0]
    for i in range(batch_q):
        for c in range(v_cols):
            acc = 0.0
            for j in range(v_rows):
                acc += weights[i, j] * v[j, c]
            result[i, c] = acc

    return result
