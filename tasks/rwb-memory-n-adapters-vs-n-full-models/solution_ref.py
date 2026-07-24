def memory_comparison(
    d: int,
    r: int,
    num_layers: int,
    N: int,
    base_params_per_layer: int,
    dtype_bytes: int,
) -> tuple[int, int]:
    base_total = base_params_per_layer * num_layers
    adapter_params = N * 2 * d * r * num_layers
    adapter_bytes = (base_total + adapter_params) * dtype_bytes
    full_bytes = base_total * N * dtype_bytes
    return (adapter_bytes, full_bytes)
