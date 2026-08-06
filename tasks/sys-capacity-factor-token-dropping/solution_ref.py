import math
import numpy as np

def token_drop_mask(assignments, num_experts, capacity_factor):
    """Return a boolean mask: True = kept, False = dropped."""
    n = len(assignments)
    capacity = math.ceil(capacity_factor * n / num_experts)
    mask = np.zeros(n, dtype=bool)
    for j in range(num_experts):
        indices = []
        for i in range(n):
            if assignments[i] == j:
                indices.append(i)
        keep = min(len(indices), capacity)
        for k in range(keep):
            mask[indices[k]] = True
    return mask
