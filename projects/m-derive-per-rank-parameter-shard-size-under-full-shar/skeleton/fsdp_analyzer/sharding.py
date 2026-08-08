def compute_rank_shard_size(num_params: int, world_size: int, rank: int) -> int:
    """Compute parameter shard size for a specific rank under FULL_SHARD."""
    raise NotImplementedError


def compute_world_shard_distribution(num_params: int, world_size: int) -> list[int]:
    """Compute list of parameter shard sizes for all ranks under FULL_SHARD."""
    raise NotImplementedError
