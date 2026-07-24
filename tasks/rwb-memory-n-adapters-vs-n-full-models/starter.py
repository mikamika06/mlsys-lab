def memory_comparison(
    d: int,
    r: int,
    num_layers: int,
    N: int,
    base_params_per_layer: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    """Return (adapter_strategy_bytes, full_copy_strategy_bytes)."""
    raise NotImplementedError("your code here")
