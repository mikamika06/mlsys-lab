def compute_costs(strategy, num_params, hidden_dim, num_layers, world_size):
    bytes_per_param = 4
    param_bytes = num_params * bytes_per_param
    if strategy == "FULL_SHARD":
        peak_memory = (param_bytes / world_size) + (2 * hidden_dim * 1024)
        call_count = 2 * num_layers
    elif strategy == "SHARD_GRAD_OP":
        peak_memory = param_bytes + (param_bytes / world_size) + (hidden_dim * 1024)
        call_count = num_layers
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return {"peak_memory": float(peak_memory), "call_count": int(call_count)}
