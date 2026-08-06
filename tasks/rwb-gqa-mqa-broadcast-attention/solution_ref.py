import math
import numpy as np


def gqa_broadcast_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    H_q, n, d = q.shape
    H_kv = k.shape[0]
    n_rep = H_q // H_kv

    k_rep = np.repeat(k, n_rep, axis=0)
    v_rep = np.repeat(v, n_rep, axis=0)

    scale = 1.0 / math.sqrt(d)
    
    scores = np.zeros((H_q, n, n), dtype=np.float64)
    for h in range(H_q):
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for l in range(d):
                    acc += q[h, i, l] * k_rep[h, j, l]
                scores[h, i, j] = acc * scale

    for h in range(H_q):
        for i in range(n):
            max_val = scores[h, i, 0]
            for j in range(1, n):
                if scores[h, i, j] > max_val:
                    max_val = scores[h, i, j]
            for j in range(n):
                scores[h, i, j] = math.exp(scores[h, i, j] - max_val)

    w = np.zeros((H_q, n, n), dtype=np.float64)
    for h in range(H_q):
        for i in range(n):
            s_sum = 0.0
            for j in range(n):
                s_sum += scores[h, i, j]
            for j in range(n):
                w[h, i, j] = scores[h, i, j] / s_sum

    out = np.zeros((H_q, n, d), dtype=np.float64)
    for h in range(H_q):
        for i in range(n):
            for l in range(d):
                acc = 0.0
                for j in range(n):
                    acc += w[h, i, j] * v_rep[h, j, l]
                out[h, i, l] = acc
    return out
