"""Sharding specifications and tensor reconstruction logic."""


def calculate_shard_spec(global_shape, world_size, rank):
    """Calculate shard shape, offset, and padding for dim 0 sharding."""
    raise NotImplementedError


def reconstruct_param(rank_shards_info):
    """Reconstruct a global parameter array from per-rank shards."""
    raise NotImplementedError
