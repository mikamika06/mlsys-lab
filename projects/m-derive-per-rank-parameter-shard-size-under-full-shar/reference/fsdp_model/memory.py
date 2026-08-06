def compute_transient_peak_memory(param_bytes: int, input_bytes: int, world_size: int) -> int:
    shard_param_bytes = (param_bytes + world_size - 1) // world_size
    unsharded_param_bytes = param_bytes
    return int(shard_param_bytes + unsharded_param_bytes + input_bytes)
