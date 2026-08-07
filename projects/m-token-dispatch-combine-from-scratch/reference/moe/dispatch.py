import numpy as np


def dispatch_tokens(tokens, indices, weights, num_experts, capacity):
    num_tokens, hidden_dim = tokens.shape
    top_k = indices.shape[1]

    expert_counts = np.zeros(num_experts, dtype=np.int64)
    dispatch_meta = {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "num_experts": num_experts,
        "capacity": capacity,
        "routes": []
    }

    expert_buffers = np.zeros((num_experts, capacity, hidden_dim), dtype=tokens.dtype)

    for t_idx in range(num_tokens):
        for k_idx in range(top_k):
            e_id = int(indices[t_idx, k_idx])
            w = float(weights[t_idx, k_idx])
            c = expert_counts[e_id]
            if c < capacity:
                expert_buffers[e_id, c] = tokens[t_idx]
                dispatch_meta["routes"].append((t_idx, k_idx, e_id, int(c), w))
                expert_counts[e_id] += 1

    return expert_buffers, dispatch_meta


def combine_tokens(expert_outputs, dispatch_meta):
    num_tokens = dispatch_meta["num_tokens"]
    hidden_dim = dispatch_meta["hidden_dim"]
    combined = np.zeros((num_tokens, hidden_dim), dtype=expert_outputs.dtype)

    for t_idx, k_idx, e_id, slot, w in dispatch_meta["routes"]:
        combined[t_idx] += w * expert_outputs[e_id, slot]

    return combined
