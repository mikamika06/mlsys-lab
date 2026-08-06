import math
import numpy as np


def reconstruct_penalty_factor(before, after, affected_indices):
    b_arr = np.asarray(before, dtype=np.float64)
    a_arr = np.asarray(after, dtype=np.float64)
    idx_arr = np.asarray(affected_indices)

    log_sum = 0.0
    count = 0
    for idx in idx_arr:
        x_val = float(b_arr[idx])
        y_val = float(a_arr[idx])
        log_sum += math.log(x_val / y_val)
        count += 1

    mean_log = log_sum / count
    return float(math.exp(mean_log))
