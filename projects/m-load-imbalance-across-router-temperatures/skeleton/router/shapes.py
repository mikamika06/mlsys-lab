def verify_alltoall_shapes(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    world_size: int,
    capacity_factor: float,
    hidden_dim: int
) -> dict:
    """Compute exact All-to-All communication shapes for Expert Parallel routing."""
    raise NotImplementedError
