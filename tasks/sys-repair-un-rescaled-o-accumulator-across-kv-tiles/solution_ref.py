import numpy as np
import math


def flash_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, tile_size: int) -> np.ndarray:
    n, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    m = [-float("inf")] * n
    l = [0.0] * n
    O = [[0.0] * d for _ in range(n)]

    K_len = K.shape[0]

    for start in range(0, K_len, tile_size):
        end = min(K_len, start + tile_size)
        tile_len = end - start

        scores = []
        for i in range(n):
            row_scores = []
            for j in range(tile_len):
                s = 0.0
                k_row = start + j
                for k in range(d):
                    s += Q[i, k] * K[k_row, k]
                row_scores.append(s * scale)
            scores.append(row_scores)

        tile_max = []
        for i in range(n):
            m_val = scores[i][0]
            for j in range(1, tile_len):
                if scores[i][j] > m_val:
                    m_val = scores[i][j]
            tile_max.append(m_val)

        new_m = []
        for i in range(n):
            if m[i] > tile_max[i]:
                new_m.append(m[i])
            else:
                new_m.append(tile_max[i])

        alpha = []
        for i in range(n):
            alpha.append(math.exp(m[i] - new_m[i]))

        exp_scores = []
        for i in range(n):
            row_exp = []
            nm = new_m[i]
            for j in range(tile_len):
                row_exp.append(math.exp(scores[i][j] - nm))
            exp_scores.append(row_exp)

        for i in range(n):
            al = alpha[i]
            nm = new_m[i]

            for c in range(d):
                v_sum = 0.0
                for j in range(tile_len):
                    v_sum += exp_scores[i][j] * V[start + j, c]
                O[i][c] = O[i][c] * al + v_sum

            sum_exp = 0.0
            for j in range(tile_len):
                sum_exp += exp_scores[i][j]
            l[i] = l[i] * al + sum_exp

            m[i] = nm

    result = np.zeros((n, d), dtype=np.float64)
    for i in range(n):
        inv_l = 1.0 / l[i]
        for c in range(d):
            result[i, c] = O[i][c] * inv_l

    return result
