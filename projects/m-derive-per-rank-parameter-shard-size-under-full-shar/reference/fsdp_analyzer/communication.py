def compute_per_step_communication_bytes(
    num_params: int,
    world_size: int,
    sharding_strategy: str,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
) -> int:
    if world_size <= 1:
        return 0

    scale = (world_size - 1) / world_size

    if sharding_strategy == "FULL_SHARD":
        param_bytes = num_params * bytes_per_param
        grad_bytes = num_params * bytes_per_grad
        return int(round(2 * scale * param_bytes + scale * grad_bytes))
    elif sharding_strategy == "SHARD_GRAD_OP":
        param_bytes = num_params * bytes_per_param
        grad_bytes = num_params * bytes_per_grad
        return int(round(scale * param_bytes + scale * grad_bytes))
    elif sharding_strategy == "NO_SHARD":
        grad_bytes = num_params * bytes_per_grad
        return int(round(scale * grad_bytes))
    else:
        raise ValueError(f"Unknown sharding strategy: {sharding_strategy}")
