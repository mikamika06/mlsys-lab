def calculate_layer_kv_bytes(layers: int, num_heads: int, head_dim: int, seq_len: int, dtype_bytes: int = 2) -> int:
    return 2 * layers * num_heads * head_dim * seq_len * dtype_bytes

def calculate_prefix_ram_bytes(model_cfg: dict, active_prefixes: list[dict], retention_hours: float) -> dict:
    layers = model_cfg["num_layers"]
    heads = model_cfg["num_kv_heads"]
    hdim = model_cfg["head_dim"]
    dtype_b = model_cfg.get("dtype_bytes", 2)
    bytes_per_token = 2 * layers * heads * hdim * dtype_b
    
    total_tokens = 0
    retained_prefixes = 0
    for p in active_prefixes:
        if p["last_accessed_hours_ago"] <= retention_hours:
            total_tokens += p["token_count"] * p.get("sharing_count", 1)
            retained_prefixes += 1
            
    base_bytes = total_tokens * bytes_per_token
    overhead_pct = model_cfg.get("overhead_ratio", 0.10)
    total_bytes = int(base_bytes * (1.0 + overhead_pct))
    
    return {
        "retained_prefixes": retained_prefixes,
        "total_tokens": total_tokens,
        "base_kv_bytes": base_bytes,
        "total_ram_bytes": total_bytes
    }
