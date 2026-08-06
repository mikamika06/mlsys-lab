def compute_kv_bytes(config, num_ctx, num_parallel):
    layers = config.get("num_hidden_layers", 32)
    kv_heads = config.get("num_key_value_heads", 8)
    head_dim = config.get("head_dim", 128)
    dtype_bytes = config.get("dtype_bytes", 2)
    bytes_per_token = layers * kv_heads * head_dim * dtype_bytes * 2
    total_tokens = num_ctx * num_parallel
    return total_tokens * bytes_per_token


def max_feasible_parallel(config, num_ctx, total_vram_bytes):
    layers = config.get("num_hidden_layers", 32)
    kv_heads = config.get("num_key_value_heads", 8)
    head_dim = config.get("head_dim", 128)
    dtype_bytes = config.get("dtype_bytes", 2)
    bytes_per_token = layers * kv_heads * head_dim * dtype_bytes * 2
    bytes_per_slot = num_ctx * bytes_per_token
    if bytes_per_slot <= 0:
        return 0
    return int(total_vram_bytes // bytes_per_slot)
