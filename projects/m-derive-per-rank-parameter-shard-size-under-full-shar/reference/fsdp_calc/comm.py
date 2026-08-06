def compute_communication_volume(total_params, world_size, strategy, bytes_per_param=2):
    param_bytes = total_params * bytes_per_param
    if strategy == "FULL_SHARD":
        return 2.0 * param_bytes * (world_size - 1) / world_size
    elif strategy == "SHARD_GRAD_OP":
        return 1.5 * param_bytes * (world_size - 1) / world_size
    elif strategy == "NO_SHARD":
        return 0.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
