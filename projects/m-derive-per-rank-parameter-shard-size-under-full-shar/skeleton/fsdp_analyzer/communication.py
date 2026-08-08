def compute_per_step_communication_bytes(
    num_params: int,
    world_size: int,
    sharding_strategy: str,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
) -> int:
    """Compute per-step communication volume in bytes across sharding strategies."""
    raise NotImplementedError
