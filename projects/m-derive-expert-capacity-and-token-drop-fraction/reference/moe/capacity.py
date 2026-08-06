import numpy as np


def compute_expert_capacity(num_tokens, num_experts, capacity_factor, top_k):
    tokens_per_expert = (num_tokens * top_k) / num_experts
    capacity = int(np.ceil(tokens_per_expert * capacity_factor))
    return max(capacity, 1)


def estimate_drop_fraction(logits, capacity_factor, top_k):
    num_tokens, num_experts = logits.shape
    capacity = compute_expert_capacity(num_tokens, num_experts, capacity_factor, top_k)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    expected_load = np.sum(probs, axis=0) * top_k
    drop_fractions = []
    for load in expected_load:
        if load <= capacity:
            drop_fractions.append(0.0)
        else:
            drop_fractions.append(float((load - capacity) / load))
    return float(np.mean(drop_fractions))
