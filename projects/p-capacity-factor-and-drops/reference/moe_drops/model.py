import numpy as np
import math

def expected_drop_rate(capacity_factor: float, num_experts: int, seq_len: int) -> float:
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

def recommend_capacity_factor(num_experts: int, seq_len: int, target_drop_rate: float) -> float:
    for cf in np.arange(1.0, 5.0, 0.05):
        if expected_drop_rate(cf, num_experts, seq_len) <= target_drop_rate:
            return float(cf)
    return 5.0
