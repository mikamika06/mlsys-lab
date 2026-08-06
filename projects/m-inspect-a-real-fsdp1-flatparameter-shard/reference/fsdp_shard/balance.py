import math

def per_rank_balance(param_sizes, world_size):
    """Compute shard sizes across ranks."""
    total = sum(param_sizes)
    padded_total = math.ceil(total / world_size) * world_size
    shard_size = padded_total // world_size
    return [shard_size for _ in range(world_size)]
