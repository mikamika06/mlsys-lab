def bytes_per_token(config: dict, dtype_bytes: int = 2) -> int:
    """Calculate bytes per token from config."""
    num_layers = config.get("num_hidden_layers", config.get("num_layers", 32))
    
    if "kv_lora_rank" in config:
        kv_lora_rank = config["kv_lora_rank"]
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        return num_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
    else:
        num_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads", 8))
        head_dim = config.get("head_dim", config.get("hidden_size", 4096) // config.get("num_attention_heads", 32))
        return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
