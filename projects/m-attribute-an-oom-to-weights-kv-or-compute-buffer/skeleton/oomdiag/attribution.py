def analyze_oom_cause(model_config, runtime_params, total_vram_bytes):
    """Diagnose memory usage and attribute OOM to weights, kv_cache, or compute_buffer."""
    raise NotImplementedError
