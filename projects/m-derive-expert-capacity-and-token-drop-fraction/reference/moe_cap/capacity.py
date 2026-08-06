import numpy as np


def compute_capacity(num_tokens, num_experts, top_k, capacity_factor):
    return int(np.ceil((num_tokens * top_k / num_experts) * capacity_factor))


def compute_drop_fraction(routes, capacity):
    num_tokens, top_k = routes.shape
    total_tokens = num_tokens * top_k
    flat = routes.flatten()
    counts = np.bincount(flat, minlength=flat.max() + 1 if flat.size > 0 else 1)
    dropped = sum(max(0, c - capacity) for c in counts)
    return float(dropped / total_tokens)
