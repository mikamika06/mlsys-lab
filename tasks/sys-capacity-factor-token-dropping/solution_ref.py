import math
import numpy as np

def token_drop_mask(assignments, num_experts, capacity_factor):
    """Return a boolean mask: True = kept, False = dropped."""
    n = len(assignments)
    capacity = math.ceil(capacity_factor * n / num_experts)
    mask = np.zeros(n, dtype=bool)
    for j in range(num_experts):
        indices = np.where(assignments == j)[0]
        keep = min(len(indices), capacity)
        mask[indices[:keep]] = True
    return mask
