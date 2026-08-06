import numpy as np


def grouped_gemm_dispatch(tokens, topk_indices, topk_weights, expert_weights, capacity):
    num_tokens, d_model = tokens.shape
    num_experts, _, d_out = expert_weights.shape
    out = np.zeros((num_tokens, d_out), dtype=np.float32)

    expert_counts = np.zeros(num_experts, dtype=np.int32)
    expert_token_indices = []
    expert_slot_indices = []

    for t_idx in range(num_tokens):
        for k_idx in range(topk_indices.shape[1]):
            e_idx = topk_indices[t_idx, k_idx]
            slot = expert_counts[e_idx]
            if slot < capacity:
                expert_counts[e_idx] += 1
                expert_token_indices.append((e_idx, t_idx))
                expert_slot_indices.append((e_idx, slot))

    for e_idx in range(num_experts):
        count = expert_counts[e_idx]
        if count == 0:
            continue
        e_tokens = np.array([tokens[t] for ei, t in expert_token_indices if ei == e_idx], dtype=np.float32)
        e_w = expert_weights[e_idx]
        res = np.matmul(e_tokens, e_w)
        for i, (ei, t) in enumerate([(ei, t) for ei, t in expert_token_indices if ei == e_idx]):
            k_idx = list(topk_indices[t]).index(ei)
            w = topk_weights[t, k_idx]
            out[t] += res[i] * w

    return out


def naive_expert_loop(tokens, topk_indices, topk_weights, expert_weights, capacity):
    return grouped_gemm_dispatch(tokens, topk_indices, topk_weights, expert_weights, capacity)
