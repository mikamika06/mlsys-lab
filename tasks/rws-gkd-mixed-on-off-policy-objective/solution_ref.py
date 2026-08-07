import math


def gkd_mixed_loss(student_logits, on_policy_targets, off_policy_targets, lam):
    x = student_logits
    on = on_policy_targets
    off = off_policy_targets

    num_rows = len(x)
    num_cols = len(x[0])

    total = 0.0
    for i in range(num_rows):
        m = x[i][0]
        for j in range(1, num_cols):
            if x[i][j] > m:
                m = x[i][j]

        s = 0.0
        for j in range(num_cols):
            s += math.exp(x[i][j] - m)

        log_z = m + math.log(s)

        on_ce = log_z - x[i][on[i]]
        off_ce = log_z - x[i][off[i]]

        total += lam * on_ce + (1.0 - lam) * off_ce

    return float(total / num_rows)
