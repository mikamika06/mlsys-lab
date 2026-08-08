def compute_rank_shard_size(num_params: int, world_size: int, rank: int) -> int:
    base = num_params // world_size
    remainder = num_params % world_size
    return base + 1 if rank < remainder else base


def compute_world_shard_distribution(num_params: int, world_size: int) -> list[int]:
    return [compute_rank_shard_size(num_params, world_size, r) for r in range(world_size)]
