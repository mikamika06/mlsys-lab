def compute_communication_volume(num_params: int, bytes_per_param: int, strategy: str, world_size: int) -> int:
    total_bytes = num_params * bytes_per_param
    if strategy == "FULL_SHARD":
        return int(2 * total_bytes * (world_size - 1) / world_size)
    elif strategy == "SHARD_GRAD_OP":
        return int(total_bytes * (world_size - 1) / world_size)
    elif strategy == "NO_SHARD":
        return 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
