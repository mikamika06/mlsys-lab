def _compute_single_offload_throughput(config, depth, ngl):
    total_layers = config["total_layers"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    element_bytes = config["element_bytes"]
    model_bytes = float(config["model_bytes"])
    gpu_bw = float(config["gpu_bw_gbps"]) * 1e9
    cpu_bw = float(config["cpu_bw_gbps"]) * 1e9
    pcie_bw = float(config["pcie_bw_gbps"]) * 1e9

    kv_bytes_per_token = float(2 * total_layers * n_kv_heads * head_dim * element_bytes)
    total_token_bytes = model_bytes + float(depth) * kv_bytes_per_token

    f_gpu = float(ngl) / float(total_layers)
    f_cpu = 1.0 - f_gpu

    gpu_bytes = f_gpu * total_token_bytes
    cpu_bytes = f_cpu * total_token_bytes

    t_gpu = gpu_bytes / gpu_bw if gpu_bw > 0 else 0.0
    t_cpu = cpu_bytes / cpu_bw if cpu_bw > 0 else 0.0

    if 0 < ngl < total_layers:
        n_heads = config["n_heads"]
        act_bytes = float(n_heads * head_dim * element_bytes)
        t_pcie = (2.0 * act_bytes) / pcie_bw if pcie_bw > 0 else 0.0
    else:
        t_pcie = 0.0

    total_time = t_gpu + t_cpu + t_pcie
    return 1.0 / total_time if total_time > 0 else 0.0


def compare_offload_throughput(config, depth, ngl1, ngl2):
    """Compare memory-bound throughput between two -ngl offload settings."""
    t1 = _compute_single_offload_throughput(config, depth, ngl1)
    t2 = _compute_single_offload_throughput(config, depth, ngl2)

    speedup = t2 / t1 if t1 > 0 else 0.0
    gain = t2 - t1

    return {
        "throughput_ngl1": t1,
        "throughput_ngl2": t2,
        "speedup": speedup,
        "offload_gain_tok_s": gain,
    }
