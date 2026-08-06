import numpy as np

CONFIGS = [
    {"num_tokens": 64, "num_experts": 8, "capacity_factor": 1.2, "top_k": 2},
    {"num_tokens": 128, "num_experts": 16, "capacity_factor": 1.0, "top_k": 2},
    {"num_tokens": 256, "num_experts": 4, "capacity_factor": 0.8, "top_k": 1},
]


def compute_expert_capacity(num_tokens, num_experts, capacity_factor, top_k):
    tokens_per_expert = (num_tokens * top_k) / num_experts
    capacity = int(np.ceil(tokens_per_expert * capacity_factor))
    return max(capacity, 1)
