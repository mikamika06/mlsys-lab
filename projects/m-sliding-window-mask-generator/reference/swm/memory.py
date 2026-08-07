def kv_cache_memory_bytes(seq_len: int, batch: int, layers: list, num_heads: int, d_head: int, dtype_bytes: int = 2) -> int:
    total_tokens = 0
    for layer in layers:
        if layer["type"] == "global":
            total_tokens += seq_len
        elif layer["type"] == "sliding":
            total_tokens += min(seq_len, layer["window_size"])

    bytes_per_token = 2 * num_heads * d_head * dtype_bytes
    return total_tokens * batch * bytes_per_token
