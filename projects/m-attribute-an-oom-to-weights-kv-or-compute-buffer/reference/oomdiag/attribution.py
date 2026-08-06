def analyze_oom_cause(model_config, runtime_params, total_vram_bytes):
    """Diagnose memory usage and attribute OOM to weights, kv_cache, or compute_buffer."""
    weights_bytes = model_config["weight_bytes"]
    
    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]
    element_size = model_config.get("kv_element_bytes", 2)
    
    ctx_size = runtime_params["n_ctx"]
    parallel = runtime_params.get("n_parallel", 1)
    
    kv_cache_bytes = 2 * n_layers * n_kv_heads * head_dim * ctx_size * parallel * element_size
    
    ubatch = runtime_params.get("n_ubatch", 512)
    hidden_dim = n_kv_heads * head_dim * model_config.get("gqa_factor", 1)
    
    compute_buffer_bytes = (
        model_config["base_graph_overhead"] + 
        ubatch * hidden_dim * 4 * 4
    )
    
    total_requested = weights_bytes + kv_cache_bytes + compute_buffer_bytes
    is_oom = total_requested > total_vram_bytes
    
    primary_culprit = None
    if is_oom:
        breakdown = {
            "weights": weights_bytes,
            "kv_cache": kv_cache_bytes,
            "compute_buffer": compute_buffer_bytes
        }
        primary_culprit = max(breakdown, key=breakdown.get)
        
    return {
        "weights_bytes": weights_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "compute_buffer_bytes": compute_buffer_bytes,
        "total_requested_bytes": total_requested,
        "is_oom": is_oom,
        "primary_culprit": primary_culprit
    }
