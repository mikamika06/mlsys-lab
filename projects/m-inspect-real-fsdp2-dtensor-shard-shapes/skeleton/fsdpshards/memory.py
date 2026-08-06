def compute_reshard_memory_profile(layer_param_shapes: list[tuple[int, ...]], mesh_size: int, dtype_bytes: int = 4, reshard_after_forward: bool = True) -> dict:
    """Measure param memory footprint during and after forward for reshard settings."""
    raise NotImplementedError
