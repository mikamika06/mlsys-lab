import numpy as np

def compare_outputs(tokens, logits, top_k, capacity_factor):
    num_tokens, num_experts = logits.shape
    capacity = int(np.ceil(capacity_factor * num_tokens * top_k / num_experts))

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    top_indices = np.argsort(probs, axis=-1)[:, -top_k:]
    top_probs = np.take_along_axis(probs, top_indices, axis=-1)

    expert_counts = np.zeros(num_experts, dtype=int)
    expert_outputs = [tokens * 1.1, tokens * 0.9, tokens * 1.2, tokens * 0.8]
    out = np.zeros_like(tokens)

    for i in range(num_tokens):
        for k_idx in range(top_k):
            exp_id = top_indices[i, k_idx]
            if expert_counts[exp_id] < capacity:
                expert_counts[exp_id] += 1
                p = top_probs[i, k_idx]
                out[i] += p * expert_outputs[exp_id][i]
    return out
