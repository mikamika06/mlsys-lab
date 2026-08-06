import math
import numpy as np


def enable_gqa_broadcast_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    batch_size, n_q, seq_q, d = Q.shape
    n_kv = K.shape[1]
    seq_k = K.shape[2]
    d_v = V.shape[-1]
    r = n_q // n_kv

    out = np.zeros((batch_size, n_q, seq_q, d_v), dtype=np.float64)
    sqrt_d = math.sqrt(d)

    for b in range(batch_size):
        for h in range(n_q):
            kv_idx = h // r
            for i in range(seq_q):
                scores = []
                for j in range(seq_k):
                    dot = 0.0
                    for k_dim in range(d):
                        dot += Q[b, h, i, k_dim] * K[b, kv_idx, j, k_dim]
                    scores.append(dot / sqrt_d)

                max_score = scores[0]
                for j in range(1, seq_k):
                    if scores[j] > max_score:
                        max_score = scores[j]

                exps = []
                sum_exp = 0.0
                for j in range(seq_k):
                    val = math.exp(scores[j] - max_score)
                    exps.append(val)
                    sum_exp += val

                weights = [val / sum_exp for val in exps]

                for v_dim in range(d_v):
                    acc = 0.0
                    for j in range(seq_k):
                        acc += weights[j] * V[b, kv_idx, j, v_dim]
                    out[b, h, i, v_dim] = acc

    return out
