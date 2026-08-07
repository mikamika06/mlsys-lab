import numpy as np


def optimal_capacity_factor(assignments, num_experts, num_tokens, max_dropped):
    if len(assignments) == 0:
        return 1.0
    counts = np.bincount(assignments[:, 1], minlength=num_experts)
    base = max(1.0, float(num_tokens) / num_experts)
    for f_int in range(10, 101):
        factor = f_int / 10.0
        cap = int(np.ceil(base * factor))
        dropped = np.sum(np.maximum(0, counts - cap))
        if dropped / len(assignments) <= max_dropped:
            return factor
    return 10.0
