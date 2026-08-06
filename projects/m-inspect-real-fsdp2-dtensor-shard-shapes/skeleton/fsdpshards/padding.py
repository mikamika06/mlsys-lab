def compare_fsdp1_fsdp2_chunking(global_shape: tuple[int, ...], mesh_size: int, dtype_bytes: int = 4) -> dict:
    """Compare FSDP1 padded sharding vs FSDP2 uneven unpadded DTensor chunking."""
    raise NotImplementedError
