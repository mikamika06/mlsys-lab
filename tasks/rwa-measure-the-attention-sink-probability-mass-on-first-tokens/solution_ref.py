import math
import numpy as np


def attention_sink_mass(logits: np.ndarray, k: int) -> float:
    x = np.asarray(logits, dtype=np.float64)
    rows = x.shape[0]
    cols = x.shape[1]

    max_vals = []
    for i in range(rows):
        m = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > m:
                m = x[i, j]
        max_vals.append(m)

    attn_rows = []
    for i in range(rows):
        row_max = max_vals[i]
        exp_row = []
        row_sum = 0.0
        for j in range(cols):
            val = math.exp(x[i, j] - row_max)
            exp_row.append(val)
            row_sum += val
        
        norm_row = [val / row_sum for val in exp_row]
        attn_rows.append(norm_row)

    column_mass = []
    for j in range(cols):
        col_sum = 0.0
        for i in range(rows):
            col_sum += attn_rows[i][j]
        column_mass.append(col_sum)

    sum_k = 0.0
    for val in column_mass[:k]:
        sum_k += val

    total_sum = 0.0
    for val in column_mass:
        total_sum += val

    return float(sum_k / total_sum)
