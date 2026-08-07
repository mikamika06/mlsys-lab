import numpy as np

def compute_dropped_fraction(expert_indices: np.ndarray, num_experts: int, capacity_factor: float) -> float:
    N = len(expert_indices)
    if N == 0:
        return 0.0
    capacity = int(np.ceil((N * capacity_factor) / num_experts))
    counts = np.bincount(expert_indices, minlength=num_experts)
    dropped = np.sum(np.maximum(counts - capacity, 0))
    return float(dropped) / N

def route_lossless(expert_indices: np.ndarray, num_experts: int):
    N = len(expert_indices)
    if N == 0:
        return 0, 0.0
    counts = np.bincount(expert_indices, minlength=num_experts)
    max_count = int(np.max(counts))
    padded_size = max_count * num_experts
    padding_fraction = (padded_size - N) / padded_size if padded_size > 0 else 0.0
    return max_count, float(padding_fraction)
