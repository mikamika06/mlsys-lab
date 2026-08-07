import numpy as np


def generate_synthetic_inputs(num_samples=200, in_dim=16, seed=42):
    np.random.seed(seed)
    return np.random.randn(num_samples, in_dim)


def get_ref_distribution(selected_experts, num_experts):
    counts = np.zeros(num_experts, dtype=np.int64)
    for idx in selected_experts.flat:
        counts[idx] += 1
    return counts


def get_ref_entropy(counts):
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def get_ref_imbalance(counts):
    max_c = np.max(counts)
    mean_c = np.mean(counts)
    if mean_c == 0:
        return 1.0
    return float(max_c / mean_c)


def get_ref_step_time(counts, capacity=100, base=1.0):
    overflow = np.maximum(0, counts - capacity)
    max_tokens = np.max(counts)
    return float(base * max_tokens + 0.05 * np.sum(overflow))
