import math
import numpy as np


def awq_vs_rtn_quality(w: np.ndarray) -> tuple[float, float]:
    """Correct implementation of AWQ vs RTN quality comparison."""
    C = w.shape[0]
    if C == 0:
        return 0.0, 0.0
    N = w.shape[1]
    rel_awq_list = []
    rel_rtn_list = []
    for i in range(C):
        row = w[i]

        max_abs = 0.0
        if N > 0:
            max_abs = abs(float(row[0]))
            for j in range(1, N):
                val_abs = abs(float(row[j]))
                if val_abs > max_abs:
                    max_abs = val_abs
        s0 = max_abs / 7.0

        q_list = []
        for j in range(N):
            val = float(row[j])
            rd = round(val / s0)
            if rd < -8:
                q_val = -8.0
            elif rd > 7:
                q_val = 7.0
            else:
                q_val = float(rd)
            q_list.append(q_val)

        sum_sq_row = 0.0
        for j in range(N):
            v = float(row[j])
            sum_sq_row += v * v
        norm_row = math.sqrt(sum_sq_row)

        sum_sq_rtn = 0.0
        for j in range(N):
            deq_rtn_j = q_list[j] * s0
            diff = deq_rtn_j - float(row[j])
            sum_sq_rtn += diff * diff
        rel_err_rtn = math.sqrt(sum_sq_rtn) / (norm_row + 1e-12)

        denom = 0.0
        for j in range(N):
            q_j = q_list[j]
            denom += q_j * q_j
        denom += 1e-12

        sum_row_q = 0.0
        for j in range(N):
            sum_row_q += float(row[j]) * q_list[j]
        s_opt = sum_row_q / denom

        sum_sq_awq = 0.0
        for j in range(N):
            deq_awq_j = q_list[j] * s_opt
            diff = deq_awq_j - float(row[j])
            sum_sq_awq += diff * diff
        rel_err_awq = math.sqrt(sum_sq_awq) / (norm_row + 1e-12)

        rel_rtn_list.append(rel_err_rtn)
        rel_awq_list.append(rel_err_awq)

    sum_awq = 0.0
    for err in rel_awq_list:
        sum_awq += err
    mean_awq = sum_awq / len(rel_awq_list)

    sum_rtn = 0.0
    for err in rel_rtn_list:
        sum_rtn += err
    mean_rtn = sum_rtn / len(rel_rtn_list)

    return mean_awq, mean_rtn
