def analyze_roofline(cfg):
    """Analyze prefill or decode roofline performance."""
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    b = cfg["batch_size"]
    s = cfg["seq_len"]
    flops = b * s * l * (12.0 * h * h + 2.0 * h * s)
    param_bytes = l * 12.0 * h * h * 2.0
    kv_bytes = 2.0 * b * s * l * (cfg["num_kv_heads"] * (h // cfg["num_heads"]) * 2.0)
    total_bytes = param_bytes + kv_bytes
    intensity = flops / total_bytes if total_bytes > 0 else 0.0
    ridge = (cfg["peak_tflops"] * 1e12) / (cfg["mem_bw_gbs"] * 1e9)
    bound = "compute" if intensity >= ridge else "memory"
    return {
        "arithmetic_intensity": float(intensity),
        "bound": bound,
        "estimated_flops": float(flops),
        "total_bytes": float(total_bytes)
    }
