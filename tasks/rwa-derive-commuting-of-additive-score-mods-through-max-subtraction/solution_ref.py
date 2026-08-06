import math
import numpy as np


def fused_attention_scores(scores, alibi, window, soft_cap):
    scores = np.asarray(scores, dtype=np.float64)
    alibi = np.asarray(alibi, dtype=np.float64)

    n = scores.shape[0]
    out_list = []

    for i in range(n):
        row_vals = []
        for j in range(n):
            val = float(scores[i, j]) + float(alibi[i, j])
            val = soft_cap * math.tanh(val / soft_cap)
            if abs(i - j) > window:
                val = float('-inf')
            row_vals.append(val)

        max_val = row_vals[0]
        for v in row_vals:
            if v > max_val:
                max_val = v

        exp_row = []
        for v in row_vals:
            if not math.isfinite(v):
                exp_row.append(0.0)
            else:
                exp_row.append(math.exp(v - max_val))

        sum_val = 0.0
        for v in exp_row:
            sum_val += v

        if sum_val == 0.0:
            norm_row = [0.0 for _ in exp_row]
        else:
            norm_row = [v / sum_val for v in exp_row]

        out_list.append(norm_row)

    return np.array(out_list, dtype=np.float64)
