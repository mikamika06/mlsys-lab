"""KV cache memory budget calculation utilities."""

BYTES_PER_ELEMENT = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
}


def compute_kv_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_len: int,
    dtype: str = "float16",
) -> int:
    """Compute exact byte budget needed for storing Key and Value cache tensors."""
    if dtype not in BYTES_PER_ELEMENT:
        raise ValueError(f"Unsupported dtype: {dtype}")
    bpe = BYTES_PER_ELEMENT[dtype]
    return 2 * num_layers * num_kv_heads * head_dim * context_len * bpe


def compute_max_context(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    byte_budget: int,
    dtype: str = "float16",
) -> int:
    """Compute maximum context sequence length fitting into the target byte budget."""
    if dtype not in BYTES_PER_ELEMENT:
        raise ValueError(f"Unsupported dtype: {dtype}")
    bpe = BYTES_PER_ELEMENT[dtype]
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bpe
    return byte_budget // bytes_per_token
