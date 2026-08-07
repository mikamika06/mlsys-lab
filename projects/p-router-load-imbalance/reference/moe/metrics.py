import numpy as np


def measure_distribution(selected_experts, num_experts):
    counts = np.zeros(num_experts, dtype=np.int64)
    for idx in selected_experts.flat:
        counts[idx] += 1
    return counts


def compute_routing_entropy(expert_counts):
    total = np.sum(expert_counts)
    if total == 0:
        return 0.0
    p = expert_counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_imbalance_ratio(expert_counts):
    max_c = np.max(expert_counts)
    mean_c = np.mean(expert_counts)
    if mean_c == 0:
        return 1.0
    return float(max_c / mean_c)
