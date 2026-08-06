def get_dtensor_shard_info(global_shape: tuple[int, ...], mesh_size: int, rank: int, shard_dim: int = 0) -> dict:
    """Calculate DTensor local shard shape and global offset for a rank."""
    raise NotImplementedError
