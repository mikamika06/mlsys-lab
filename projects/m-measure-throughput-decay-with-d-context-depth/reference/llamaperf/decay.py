def measure_context_decay(config, depths):
    """Calculate context decay metrics for a given model config and depth list."""
    if not depths:
        return {"kv_bytes_per_token": 0.0, "throughputs": [], "decay_ratios": []}

    total_layers = config["total_layers"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    element_bytes = config["element_bytes"]
    model_bytes = float(config["model_bytes"])
    gpu_bw_bytes_sec = float(config["gpu_bw_gbps"]) * 1e9

    kv_bytes_per_token = float(2 * total_layers * n_kv_heads * head_dim * element_bytes)

    throughputs = []
    for d in depths:
        total_bytes = model_bytes + float(d) * kv_bytes_per_token
        t_sec = total_bytes / gpu_bw_bytes_sec
        throughputs.append(1.0 / t_sec)

    base_tp = throughputs[0]
    decay_ratios = [tp / base_tp for tp in throughputs]

    return {
        "kv_bytes_per_token": kv_bytes_per_token,
        "throughputs": throughputs,
        "decay_ratios": decay_ratios,
    }
