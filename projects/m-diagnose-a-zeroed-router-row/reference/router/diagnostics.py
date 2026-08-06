import numpy as np


def diagnose_router_matrix(router_weights, top_k, max_capacity=None):
    weights = np.asarray(router_weights, dtype=np.float64)
    num_tokens, num_experts = weights.shape

    row_sums = np.sum(weights, axis=1)
    zero_rows = np.where(row_sums == 0.0)[0].tolist()

    non_zero_mask = row_sums > 0.0
    top_k_indices = np.argsort(-weights, axis=1)[:, :top_k]

    active_mask = np.zeros_like(weights, dtype=bool)
    for i in range(num_tokens):
        if non_zero_mask[i]:
            for k_idx in range(top_k):
                if weights[i, top_k_indices[i, k_idx]] > 0.0:
                    active_mask[i, top_k_indices[i, k_idx]] = True

    expert_counts = np.sum(active_mask, axis=0).tolist()

    dropped_tokens = 0
    if max_capacity is not None:
        for e in range(num_experts):
            if expert_counts[e] > max_capacity:
                dropped_tokens += expert_counts[e] - max_capacity

    unassigned_tokens = [i for i in range(num_tokens) if not np.any(active_mask[i])]

    return {
        "zero_rows": zero_rows,
        "unassigned_tokens": unassigned_tokens,
        "expert_counts": expert_counts,
        "dropped_tokens": dropped_tokens
    }


def count_moe_parameters(config):
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    num_layers = config.get("num_layers", 1)
    shared_expert = config.get("has_shared_expert", False)

    expert_gate_proj = hidden_size * intermediate_size
    expert_up_proj = hidden_size * intermediate_size
    expert_down_proj = intermediate_size * hidden_size
    per_expert_params = expert_gate_proj + expert_up_proj + expert_down_proj

    router_params = hidden_size * num_experts

    total_layer_params = (num_experts * per_expert_params) + router_params
    active_layer_params = (top_k * per_expert_params) + router_params

    if shared_expert:
        total_layer_params += per_expert_params
        active_layer_params += per_expert_params

    return {
        "total_parameters": int(total_layer_params * num_layers),
        "active_parameters": int(active_layer_params * num_layers),
        "router_parameters": int(router_params * num_layers),
        "expert_parameters": int(num_experts * per_expert_params * num_layers)
    }
