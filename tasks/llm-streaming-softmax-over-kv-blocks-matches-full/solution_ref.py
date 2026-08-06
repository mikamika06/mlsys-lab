import math
import numpy as np


def streaming_softmax_attention(Q, K, V, block_size):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    v_dim = V.shape[1]
    scale = 1.0 / math.sqrt(d)
    out = np.empty((n, v_dim), dtype=np.float64)

    for i in range(n):
        m = -math.inf
        l = 0.0
        o = np.zeros(v_dim, dtype=np.float64)

        for start in range(0, n, block_size):
            end = start + block_size
            if end > n:
                end = n

            block_len = end - start
            scores = [0.0] * block_len

            for b in range(block_len):
                j = start + b
                dot_val = 0.0
                for k in range(d):
                    dot_val += float(Q[i, k]) * float(K[j, k])
                scores[b] = dot_val * scale

            block_max = -math.inf
            for b in range(block_len):
                if scores[b] > block_max:
                    block_max = scores[b]

            new_m = m if m > block_max else block_max

            old_scale = 0.0 if l == 0.0 else l * math.exp(m - new_m)

            weights = [0.0] * block_len
            sum_weights = 0.0
            for b in range(block_len):
                w = math.exp(scores[b] - new_m)
                weights[b] = w
                sum_weights += w

            new_l = old_scale + sum_weights

            weights_V = [0.0] * v_dim
            for c in range(v_dim):
                v_sum = 0.0
                for b in range(block_len):
                    j = start + b
                    v_sum += weights[b] * float(V[j, c])
                weights_V[c] = v_sum

            for c in range(v_dim):
                o[c] = (old_scale * float(o[c]) + weights_V[c]) / new_l

            m = new_m
            l = new_l

        out[i] = o

    return out
