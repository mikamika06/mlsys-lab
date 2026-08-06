def compute_kv_bytes(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
) -> int:
    """Compute theoretical KV cache byte size for f16, q8_0, and q4_0."""
    if head_dim % 32 != 0:
        raise ValueError("head_dim must be a multiple of 32")
    if dtype not in ("f16", "q8_0", "q4_0"):
        raise ValueError(f"Unsupported dtype: {dtype}")

    total_elements = num_tokens * 2 * num_layers * num_kv_heads * head_dim
    num_blocks = total_elements // 32

    if dtype == "f16":
        return total_elements * 2
    elif dtype == "q8_0":
        return num_blocks * 34
    else:
        return num_blocks * 18
