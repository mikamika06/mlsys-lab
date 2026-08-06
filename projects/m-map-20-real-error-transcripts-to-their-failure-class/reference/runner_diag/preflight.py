def predict_fit(model_cfg: dict, ctx_size: int, parallel_cfg: dict, sys_state: dict) -> dict:
    tp = parallel_cfg.get("tensor_parallel", 1)
    pp = parallel_cfg.get("pipeline_parallel", 1)

    num_layers = model_cfg["num_layers"]
    hidden_size = model_cfg["hidden_size"]
    num_heads = model_cfg["num_attention_heads"]
    num_kv_heads = model_cfg.get("num_key_value_heads", num_heads)
    bytes_per_param = model_cfg.get("bytes_per_param", 2.0)

    num_params = model_cfg.get("num_params", 7000000000)

    weight_vram_mb = (num_params * bytes_per_param) / (1024 * 1024) / (tp * pp)

    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim
    kv_cache_bytes_per_token = 2 * num_layers * kv_dim * bytes_per_param
    total_kv_vram_mb = (kv_cache_bytes_per_token * ctx_size) / (1024 * 1024) / tp

    act_vram_mb = (ctx_size * hidden_size * 4 * bytes_per_param) / (1024 * 1024) / tp

    runtime_overhead_vram_mb = 512.0

    req_vram_mb = weight_vram_mb + total_kv_vram_mb + act_vram_mb + runtime_overhead_vram_mb
    req_ram_mb = 2048.0

    free_vram = sys_state.get("free_vram_mb", 0.0)
    free_ram = sys_state.get("free_ram_mb", 0.0)

    fits = (req_vram_mb <= free_vram) and (req_ram_mb <= free_ram)

    bottleneck = None
    if not fits:
        if req_vram_mb > free_vram:
            bottleneck = "VRAM"
        elif req_ram_mb > free_ram:
            bottleneck = "RAM"

    return {
        "fits": fits,
        "required_vram_mb": round(req_vram_mb, 2),
        "required_ram_mb": round(req_ram_mb, 2),
        "bottleneck": bottleneck
    }
