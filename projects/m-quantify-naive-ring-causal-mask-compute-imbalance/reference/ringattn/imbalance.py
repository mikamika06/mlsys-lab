import numpy as np


def compute_imbalance(seq_len, num_ranks):
    block_size = seq_len // num_ranks
    workloads = []
    for r in range(num_ranks):
        active_blocks = 0
        for step in range(num_ranks):
            k_block = (r - step) % num_ranks
            if k_block <= r:
                active_blocks += 1
        workloads.append(active_blocks)
    max_w = float(max(workloads))
    avg_w = float(sum(workloads)) / float(num_ranks)
    imbalance_ratio = max_w / avg_w if avg_w > 0 else 1.0
    return {
        "workloads": workloads,
        "max_workload": max_w,
        "avg_workload": avg_w,
        "imbalance_ratio": imbalance_ratio,
    }
