import numpy as np


def measure_drop_rates(logits, capacity_factors, top_k):
    num_tokens, num_experts = logits.shape
    top_indices = np.argsort(logits, axis=1)[:, -top_k:]
    results = []
    for cf in capacity_factors:
        cap = int(np.ceil((num_tokens * top_k / num_experts) * cf))
        flat = top_indices.flatten()
        counts = np.bincount(flat, minlength=num_experts)
        dropped = sum(max(0, c - cap) for c in counts)
        results.append(float(dropped / (num_tokens * top_k)))
    return results
