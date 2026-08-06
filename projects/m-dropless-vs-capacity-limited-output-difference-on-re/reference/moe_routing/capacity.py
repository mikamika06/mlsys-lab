import numpy as np

def select_capacity_factor(tokens, logits, top_k, max_drop_rate):
    factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    num_tokens, num_experts = logits.shape
    total_assignments = num_tokens * top_k

    for cf in factors:
        capacity = int(np.ceil(cf * total_assignments / num_experts))
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        top_indices = np.argsort(probs, axis=-1)[:, -top_k:]

        expert_counts = np.zeros(num_experts, dtype=int)
        dropped = 0
        for i in range(num_tokens):
            for k_idx in range(top_k):
                exp_id = top_indices[i, k_idx]
                if expert_counts[exp_id] < capacity:
                    expert_counts[exp_id] += 1
                else:
                    dropped += 1
        drop_rate = dropped / total_assignments
        if drop_rate <= max_drop_rate:
            return cf
    return 2.0
