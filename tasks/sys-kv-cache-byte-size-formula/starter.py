def kv_cache_size(L: int, n_q: int, n_kv: int, d: int, seq: int,
                   dtype_bytes: int):
    """
    Return (kv_bytes, ratio_vs_mha):
      kv_bytes = 2 * L * n_kv * d * seq * dtype_bytes -- total K+V cache
        bytes for this config.
      ratio_vs_mha = kv_bytes(n_kv) / kv_bytes(n_q) -- the ratio against
        the full-MHA baseline (n_kv == n_q) at the same L, d, seq,
        dtype_bytes; should equal n_kv / n_q.
    """
    raise NotImplementedError('your code here')
