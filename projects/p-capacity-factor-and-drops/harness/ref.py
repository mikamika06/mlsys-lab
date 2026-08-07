import numpy as np
import math

def get_indices(seed: int, N: int, E: int):
    rng = np.random.default_rng(seed)
    return rng.integers(0, E, size=N)

def ref_compute_dropped_fraction(expert_indices, num_experts, capacity_factor):
    N = len(expert_indices)
    if N == 0:
        return 0.0
    capacity = int(np.ceil((N * capacity_factor) / num_experts))
    counts = np.bincount(expert_indices, minlength=num_experts)
    dropped = np.sum(np.maximum(counts - capacity, 0))
    return float(dropped) / N

def ref_route_lossless(expert_indices, num_experts):
    N = len(expert_indices)
    if N == 0:
        return 0, 0.0
    counts = np.bincount(expert_indices, minlength=num_experts)
    max_count = int(np.max(counts))
    padded_size = max_count * num_experts
    padding_fraction = (padded_size - N) / padded_size if padded_size > 0 else 0.0
    return max_count, float(padding_fraction)

def ref_expected_drop_rate(capacity_factor, num_experts, seq_len):
    if seq_len == 0:
        return 0.0
    capacity = int(np.ceil((seq_len * capacity_factor) / num_experts))
    lam = seq_len / num_experts
    expected_drops = 0.0
    bound = int(lam + 10 * math.sqrt(lam)) + 10
    for k in range(capacity + 1, bound):
        log_pmf = -lam + k * math.log(lam) - math.lgamma(k + 1)
        expected_drops += (k - capacity) * math.exp(log_pmf)
    return min((expected_drops * num_experts) / seq_len, 1.0)

def ref_quality_penalty(drop_rate):
    return 1.0 - (1.0 - drop_rate)**3

def ref_recommend_capacity_factor(num_experts, seq_len, target_drop_rate):
    for cf in np.arange(1.0, 5.0, 0.05):
        if ref_expected_drop_rate(cf, num_experts, seq_len) <= target_drop_rate:
            return float(cf)
    return 5.0
