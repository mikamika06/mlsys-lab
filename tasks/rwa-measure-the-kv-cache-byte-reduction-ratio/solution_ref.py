def kv_cache_stats(
    n_kv_heads: int,
    n_q_heads: int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> tuple[int, int, float]:
    """
    Return the incremental byte counts for a KV cache and a full MHA,
    together with their ratio.
    """
    kv_bytes = 2 * n_kv_heads * head_dim * dtype_bytes
    mha_bytes = 2 * n_q_heads * head_dim * dtype_bytes
    return kv_bytes, mha_bytes, kv_bytes / mha_bytes
