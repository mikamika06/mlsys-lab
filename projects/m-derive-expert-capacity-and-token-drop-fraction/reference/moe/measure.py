import numpy as np
from moe.capacity import compute_expert_capacity


def measure_drop_rate(router_logits, capacity_factors, top_k):
    num_tokens, num_experts = router_logits.shape
    results = []
    topk_indices = np.argsort(router_logits, axis=-1)[:, -top_k:]
    for cf in capacity_factors:
        capacity = compute_expert_capacity(num_tokens, num_experts, cf, top_k)
        expert_counts = np.zeros(num_experts, dtype=np.int32)
        dropped = 0
        total = num_tokens * top_k
        for t in range(num_tokens):
            for k in range(top_k):
                e = topk_indices[t, k]
                if expert_counts[e] < capacity:
                    expert_counts[e] += 1
                else:
                    dropped += 1
        results.append(float(dropped / total))
    return results
