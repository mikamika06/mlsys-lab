def get_dtensor_shard_info(global_shape: tuple[int, ...], mesh_size: int, rank: int, shard_dim: int = 0) -> dict:
    """Calculate DTensor local shard shape and global offset for a rank."""
    if rank < 0 or rank >= mesh_size:
        raise ValueError("Invalid rank")

    dim_size = global_shape[shard_dim]
    base = dim_size // mesh_size
    rem = dim_size % mesh_size

    if rank < rem:
        local_dim = base + 1
        offset_dim = rank * (base + 1)
    else:
        local_dim = base
        offset_dim = rem * (base + 1) + (rank - rem) * base

    local_shape = list(global_shape)
    local_shape[shard_dim] = local_dim

    offset = [0] * len(global_shape)
    offset[shard_dim] = offset_dim

    numel_global = 1
    for s in global_shape:
        numel_global *= s

    numel_local = 1
    for s in local_shape:
        numel_local *= s

    return {
        "global_shape": tuple(global_shape),
        "local_shape": tuple(local_shape),
        "offset": tuple(offset),
        "numel_local": numel_local,
        "numel_global": numel_global,
    }
