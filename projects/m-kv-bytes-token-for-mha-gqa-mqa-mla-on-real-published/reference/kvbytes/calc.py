def calc_bytes_per_token(config, dtype_bytes=2):
    layers = config.get("num_layers", 1)
    attn_type = config.get("attn_type", "mha")
    num_heads = config.get("num_heads", 32)
    num_kv_heads = config.get("num_kv_heads", num_heads)
    head_dim = config.get("head_dim", 128)
    if attn_type == "mha":
        kv_heads = num_heads
    elif attn_type in ("gqa", "mqa"):
        kv_heads = num_kv_heads
    else:
        kv_heads = num_kv_heads
    return 2 * layers * kv_heads * head_dim * dtype_bytes
