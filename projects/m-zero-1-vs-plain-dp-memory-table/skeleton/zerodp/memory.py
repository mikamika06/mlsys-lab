def calc_memory_table(
    param_counts: list[int],
    world_size: int,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
    opt_bytes_per_param: int = 12,
) -> dict:
    """Calculate memory breakdown for Plain DP vs ZeRO Stage 1."""
    raise NotImplementedError
