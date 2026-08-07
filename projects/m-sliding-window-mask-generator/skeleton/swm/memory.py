def kv_cache_memory_bytes(seq_len: int, batch: int, layers: list, num_heads: int, d_head: int, dtype_bytes: int = 2) -> int:
    """
    Calculate the total KV cache size in bytes for a hybrid model.
    `layers` is a list of dicts:
       {"type": "global"} OR {"type": "sliding", "window_size": int}
    """
    raise NotImplementedError
