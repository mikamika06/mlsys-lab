import ref
import numpy as np


def check(workdir):
    from matmul_acc.scheduling import compute_l2_hit_rate
    rate_row = compute_l2_hit_rate(2048, 2048, 2048, 64, 64, grouped=False)
    rate_group = compute_l2_hit_rate(2048, 2048, 2048, 64, 64, grouped=True)
    improvement = rate_group / (rate_row + 1e-7)
    return {"grouped_improvement": float(improvement)}
