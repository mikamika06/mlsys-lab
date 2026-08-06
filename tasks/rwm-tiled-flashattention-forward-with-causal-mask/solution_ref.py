import math
import numpy as np


def flash_attention_forward(Q, K, V, block_size=2):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    n, d = Q.shape
    scale = 1.0 / math.sqrt(float(d))
    O = np.zeros((n, d), dtype=np.float64)

    for qs in range(0, n, block_size):
        qe = qs + block_size
        if qe > n:
            qe = n

        rows = qe - qs
        m = [-float("inf")] * rows
        l = [0.0] * rows
        acc = [[0.0] * d for _ in range(rows)]

        for ks in range(0, n, block_size):
            ke = ks + block_size
            if ke > n:
                ke = n

            cols = ke - ks

            scores = [[0.0] * cols for _ in range(rows)]
            for i in range(rows):
                r_idx = qs + i
                for j in range(cols):
                    c_idx = ks + j
                    if c_idx > r_idx:
                        scores[i][j] = -float("inf")
                    else:
                        dot = 0.0
                        for k_dim in range(d):
                            dot += Q[r_idx, k_dim] * K[c_idx, k_dim]
                        scores[i][j] = dot * scale

            block_max = [-float("inf")] * rows
            for i in range(rows):
                b_max = -float("inf")
                for j in range(cols):
                    val = scores[i][j]
                    if val > b_max:
                        b_max = val
                block_max[i] = b_max

            new_m = [0.0] * rows
            for i in range(rows):
                if m[i] > block_max[i]:
                    new_m[i] = m[i]
                else:
                    new_m[i] = block_max[i]

            old_scale = [0.0] * rows
            for i in range(rows):
                old_scale[i] = math.exp(m[i] - new_m[i])

            exp_scores = [[0.0] * cols for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    exp_scores[i][j] = math.exp(scores[i][j] - new_m[i])

            new_l = [0.0] * rows
            for i in range(rows):
                s_sum = 0.0
                for j in range(cols):
                    s_sum += exp_scores[i][j]
                new_l[i] = old_scale[i] * l[i] + s_sum
            l = new_l

            new_acc = [[0.0] * d for _ in range(rows)]
            for i in range(rows):
                for k_dim in range(d):
                    matmul_val = 0.0
                    for j in range(cols):
                        matmul_val += exp_scores[i][j] * V[ks + j, k_dim]
                    new_acc[i][k_dim] = old_scale[i] * acc[i][k_dim] + matmul_val
            acc = new_acc

            m = new_m

        for i in range(rows):
            inv_l = 1.0 / l[i]
            for k_dim in range(d):
                O[qs + i, k_dim] = acc[i][k_dim] * inv_l

    return O
