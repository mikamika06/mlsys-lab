import math
import numpy as np


def chunked_attention(q, chunks):
    q = np.asarray(q, dtype=np.float64)
    m = -float("inf")
    l = 0.0
    out = np.zeros(chunks[0][1].shape[1], dtype=np.float64)

    for K, V in chunks:
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        
        n_rows = K.shape[0]
        dim_q = q.shape[0]
        scores = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            dot_val = 0.0
            for j in range(dim_q):
                dot_val += K[i, j] * q[j]
            scores[i] = dot_val

        chunk_max = -float("inf")
        for i in range(n_rows):
            if scores[i] > chunk_max:
                chunk_max = scores[i]

        new_m = m if m > chunk_max else chunk_max

        old_scale = 0.0 if m == -float("inf") else math.exp(m - new_m)
        
        weights = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            weights[i] = math.exp(scores[i] - new_m)

        dim_v = V.shape[1]
        weights_V = np.zeros(dim_v, dtype=np.float64)
        for j in range(dim_v):
            s = 0.0
            for i in range(n_rows):
                s += weights[i] * V[i, j]
            weights_V[j] = s

        scale_factor = l * old_scale
        for j in range(dim_v):
            out[j] = out[j] * scale_factor + weights_V[j]

        weights_sum = 0.0
        for i in range(n_rows):
            weights_sum += weights[i]

        l = scale_factor + weights_sum
        m = new_m
        
        for j in range(dim_v):
            out[j] = out[j] / l

    return out
