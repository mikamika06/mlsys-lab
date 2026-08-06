import numpy as np


def naive_per_expert_loop(tokens, routes, weights, capacity):
    num_tokens, hidden_dim = tokens.shape
    num_experts = routes.max() + 1
    out = np.zeros_like(tokens, dtype=np.float32)
    for e in range(num_experts):
        matches = np.where(routes == e)
        token_indices = matches[0]
        if len(token_indices) == 0:
            continue
        if len(token_indices) > capacity:
            token_indices = token_indices[:capacity]
        selected = tokens[token_indices]
        out[token_indices] += selected * 1.0
    return out


def grouped_gemm_dispatch(tokens, routes, weights, capacity):
    return naive_per_expert_loop(tokens, routes, weights, capacity)
