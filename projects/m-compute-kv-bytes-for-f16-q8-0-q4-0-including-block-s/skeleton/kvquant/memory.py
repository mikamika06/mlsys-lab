def measure_kv_footprint(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
) -> dict:
    """Measure allocated memory footprint for KV cache buffers."""
    raise NotImplementedError
