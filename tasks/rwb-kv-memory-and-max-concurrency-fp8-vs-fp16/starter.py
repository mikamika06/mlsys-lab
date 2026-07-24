def kv_capacity(config, vram_budget_bytes):
    """KV bytes/token and max concurrent sequences at fp8 vs fp16.

    config: dict with num_layers, num_kv_heads, head_dim, seq_len (ints).
    vram_budget_bytes: positive int, VRAM budget for the KV cache.

    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
      (bytes_per_elem = 1 for fp8, 2 for fp16)
    max_concurrent_sequences = floor(vram_budget_bytes / (bytes_per_token * seq_len))

    Returns a dict with keys: bytes_per_token_fp8, bytes_per_token_fp16,
    max_concurrent_fp8, max_concurrent_fp16.
    """
    raise NotImplementedError('your code here')
