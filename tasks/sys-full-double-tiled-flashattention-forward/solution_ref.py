import math
import numpy as np


def flash_attention_forward(Q, K, V, Br, Bc):
    n = Q.shape[0]
    d = Q.shape[1]
    dv = V.shape[1]
    scale = 1.0 / math.sqrt(d)

    output = np.zeros((n, dv), dtype=np.float64)

    for row_start in range(0, n, Br):
        row_end = min(row_start + Br, n)
        Br_actual = row_end - row_start

        running_max = np.full((Br_actual,), -float("inf"), dtype=np.float64)
        running_sum = np.zeros((Br_actual,), dtype=np.float64)
        running_output = np.zeros((Br_actual, dv), dtype=np.float64)

        for col_start in range(0, n, Bc):
            col_end = min(col_start + Bc, n)
            Bc_actual = col_end - col_start

            score_tile = np.zeros((Br_actual, Bc_actual), dtype=np.float64)
            for i in range(Br_actual):
                q_row = row_start + i
                for j in range(Bc_actual):
                    k_row = col_start + j
                    dot_val = 0.0
                    for k in range(d):
                        dot_val += Q[q_row, k] * K[k_row, k]
                    score_tile[i, j] = dot_val * scale

            tile_max = np.zeros((Br_actual,), dtype=np.float64)
            for i in range(Br_actual):
                m_val = -float("inf")
                for j in range(Bc_actual):
                    val = score_tile[i, j]
                    if val > m_val:
                        m_val = val
                tile_max[i] = m_val

            new_max = np.zeros((Br_actual,), dtype=np.float64)
            for i in range(Br_actual):
                if running_max[i] > tile_max[i]:
                    new_max[i] = running_max[i]
                else:
                    new_max[i] = tile_max[i]

            old_factor = np.zeros((Br_actual,), dtype=np.float64)
            for i in range(Br_actual):
                old_factor[i] = math.exp(running_max[i] - new_max[i])

            exp_tile = np.zeros((Br_actual, Bc_actual), dtype=np.float64)
            for i in range(Br_actual):
                for j in range(Bc_actual):
                    exp_tile[i, j] = math.exp(score_tile[i, j] - new_max[i])

            sum_exp = np.zeros((Br_actual,), dtype=np.float64)
            for i in range(Br_actual):
                s_val = 0.0
                for j in range(Bc_actual):
                    s_val += exp_tile[i, j]
                sum_exp[i] = s_val

            new_sum = np.zeros((Br_actual,), dtype=np.float64)
            for i in range(Br_actual):
                new_sum[i] = old_factor[i] * running_sum[i] + sum_exp[i]

            matmul_res = np.zeros((Br_actual, dv), dtype=np.float64)
            for i in range(Br_actual):
                for l in range(dv):
                    dot_v = 0.0
                    for j in range(Bc_actual):
                        dot_v += exp_tile[i, j] * V[col_start + j, l]
                    matmul_res[i, l] = dot_v

            numerator = np.zeros((Br_actual, dv), dtype=np.float64)
            for i in range(Br_actual):
                term1 = old_factor[i] * running_sum[i]
                for l in range(dv):
                    numerator[i, l] = term1 * running_output[i, l] + matmul_res[i, l]

            for i in range(Br_actual):
                ns = new_sum[i]
                for l in range(dv):
                    running_output[i, l] = numerator[i, l] / ns

            for i in range(Br_actual):
                running_max[i] = new_max[i]
                running_sum[i] = new_sum[i]

        for i in range(Br_actual):
            for l in range(dv):
                output[row_start + i, l] = running_output[i, l]

    return output
