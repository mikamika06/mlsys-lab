def compute_kv_bytes(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
) -> int:
    """Compute theoretical KV cache byte size for f16, q8_0, and q4_0."""
    raise NotImplementedError
