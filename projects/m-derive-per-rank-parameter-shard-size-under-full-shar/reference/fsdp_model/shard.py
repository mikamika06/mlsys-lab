def compute_shard_sizes(total_params: int, world_size: int) -> list[int]:
    base = total_params // world_size
    rem = total_params % world_size
    sizes = []
    for i in range(world_size):
        sizes.append(base + (1 if i < rem else 0))
    return sizes
