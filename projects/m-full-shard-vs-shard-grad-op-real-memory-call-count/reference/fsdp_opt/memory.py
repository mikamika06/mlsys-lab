def calculate_memory_and_calls(strategy, num_params, param_bytes, world_size):
    if strategy == "FULL_SHARD":
        param_mem = (num_params * param_bytes) / world_size
        grad_mem = (num_params * param_bytes) / world_size
        all_gather_calls = 2
        reduce_scatter_calls = 1
    elif strategy == "SHARD_GRAD_OP":
        param_mem = num_params * param_bytes
        grad_mem = (num_params * param_bytes) / world_size
        all_gather_calls = 0
        reduce_scatter_calls = 1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return float(param_mem), float(grad_mem), int(all_gather_calls), int(reduce_scatter_calls)
