def mla_bytes_per_token(config, dtype_bytes=2):
    layers = config.get("num_layers", 1)
    kv_lora_rank = config.get("kv_lora_rank", 512)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
    num_heads = config.get("num_heads", 128)
    return layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
