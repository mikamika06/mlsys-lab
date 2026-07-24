def kv_cache_transfer_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    seq_len: int
) -> int:
    """
    Compute the number of bytes that must be transferred for a KV cache request.
    All arguments are integers; the result is an integer as well.
    """
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
