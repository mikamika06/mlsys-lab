def kv_cache_size(L: int, n_q: int, n_kv: int, d: int, seq: int,
                   dtype_bytes: int):
    """
    Total KV-cache size in bytes for this config, and its ratio against the
    full-MHA (n_kv == n_q) baseline at the same L, d, seq, dtype_bytes.
    """
    kv_bytes = 2 * L * n_kv * d * seq * dtype_bytes
    mha_bytes = 2 * L * n_q * d * seq * dtype_bytes
    ratio_vs_mha = kv_bytes / mha_bytes
    return kv_bytes, ratio_vs_mha
