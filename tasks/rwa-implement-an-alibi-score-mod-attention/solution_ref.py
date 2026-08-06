import math
import numpy as np


def alibi_score_mod_attention(Q, K, V, slopes):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)

    H, n, d = Q.shape
    m = K.shape[1]
    dv = V.shape[2]

    scores = np.zeros((H, n, m), dtype=np.float64)
    scale = 1.0 / math.sqrt(d)

    for h in range(H):
        for i in range(n):
            for j in range(m):
                dot_val = 0.0
                for k_d in range(d):
                    dot_val += Q[h, i, k_d] * K[h, j, k_d]
                bias_val = float(j) - float(i)
                scores[h, i, j] = (dot_val * scale) + (slopes[h] * bias_val)

    for h in range(H):
        for i in range(n):
            max_val = scores[h, i, 0]
            for j in range(1, m):
                if scores[h, i, j] > max_val:
                    max_val = scores[h, i, j]
            for j in range(m):
                scores[h, i, j] = math.exp(scores[h, i, j] - max_val)

    weights = np.zeros((H, n, m), dtype=np.float64)
    for h in range(H):
        for i in range(n):
            sum_val = 0.0
            for j in range(m):
                sum_val += scores[h, i, j]
            for j in range(m):
                weights[h, i, j] = scores[h, i, j] / sum_val

    out = np.zeros((H, n, dv), dtype=np.float64)
    for h in range(H):
        for i in range(n):
            for v_idx in range(dv):
                acc = 0.0
                for j in range(m):
                    acc += weights[h, i, j] * V[h, j, v_idx]
                out[h, i, v_idx] = acc

    return out
