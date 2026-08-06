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
    raise NotImplementedError


def compute_max_context(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    byte_budget: int,
    dtype: str = "float16",
) -> int:
    """Compute maximum context sequence length fitting into the target byte budget."""
    raise NotImplementedError
