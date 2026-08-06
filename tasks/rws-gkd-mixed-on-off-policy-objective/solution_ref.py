import math
import numpy as np


def gkd_mixed_loss(student_logits, on_policy_targets, off_policy_targets, lam):
    x = np.asarray(student_logits, dtype=np.float64)
    on = np.asarray(on_policy_targets, dtype=np.int64)
    off = np.asarray(off_policy_targets, dtype=np.int64)

    num_rows = x.shape[0]
    num_cols = x.shape[1]

    total = 0.0
    for i in range(num_rows):
        m = x[i, 0]
        for j in range(1, num_cols):
            if x[i, j] > m:
                m = x[i, j]

        s = 0.0
        for j in range(num_cols):
            s += math.exp(x[i, j] - m)

        log_z = m + math.log(s)

        on_ce = log_z - x[i, on[i]]
        off_ce = log_z - x[i, off[i]]

        total += lam * on_ce + (1.0 - lam) * off_ce

    return float(total / num_rows)
