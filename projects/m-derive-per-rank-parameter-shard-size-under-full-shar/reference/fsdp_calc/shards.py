def compute_shard_sizes(total_params, world_size):
    base = total_params // world_size
    rem = total_params % world_size
    sizes = []
    for r in range(world_size):
        sizes.append(base + (1 if r < rem else 0))
    return sizes
