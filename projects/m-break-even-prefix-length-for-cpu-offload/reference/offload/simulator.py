def compute_kv_bytes(config, prefix_len):
    """Calculate KV cache memory usage in bytes for a given prefix length."""
    return 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"] * prefix_len


def estimate_recompute_time(config, hw, prefix_len):
    """Estimate prompt recomputation latency on GPU in seconds."""
    flops = 2 * config["num_params"] * prefix_len
    gpu_flops = hw["gpu_tflops"] * 1e12
    return (flops / gpu_flops) + hw.get("launch_overhead_s", 0.0)


def estimate_load_time(config, tier, prefix_len):
    """Estimate offload retrieval latency from a storage tier in seconds."""
    kv_bytes = compute_kv_bytes(config, prefix_len)
    bw_bps = tier["bandwidth_gbps"] * 1e9
    return (kv_bytes / bw_bps) + tier.get("latency_s", 0.0)


def compute_breakeven_prefix_length(config, hw, tier):
    """Compute exact crossover prefix length where load time equals recompute time."""
    c_comp = (2 * config["num_params"]) / (hw["gpu_tflops"] * 1e12)
    o_comp = hw.get("launch_overhead_s", 0.0)

    bytes_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"]
    c_trans = bytes_per_token / (tier["bandwidth_gbps"] * 1e9)
    o_trans = tier.get("latency_s", 0.0)

    denom = c_comp - c_trans
    if abs(denom) < 1e-12:
        return float("inf") if o_trans > o_comp else 0.0

    l_be = (o_trans - o_comp) / denom
    if l_be < 0:
        return float("inf") if c_trans >= c_comp else 0.0
    return l_be
