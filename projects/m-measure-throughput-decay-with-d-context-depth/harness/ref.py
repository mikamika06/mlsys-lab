CONFIGS = [
    {
        "total_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "element_bytes": 2,
        "model_bytes": 7_000_000_000,
        "gpu_bw_gbps": 900.0,
        "cpu_bw_gbps": 60.0,
        "pcie_bw_gbps": 32.0,
    },
    {
        "total_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 40,
        "head_dim": 128,
        "element_bytes": 2,
        "model_bytes": 13_000_000_000,
        "gpu_bw_gbps": 1000.0,
        "cpu_bw_gbps": 80.0,
        "pcie_bw_gbps": 32.0,
    },
    {
        "total_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,
        "head_dim": 128,
        "element_bytes": 2,
        "model_bytes": 70_000_000_000,
        "gpu_bw_gbps": 2000.0,
        "cpu_bw_gbps": 100.0,
        "pcie_bw_gbps": 64.0,
    },
]

DEPTH_SETS = [
    [512, 1024, 2048, 4096, 8192],
    [1024, 4096, 16384, 32768],
    [2048, 8192, 32768, 65536],
]

OFFLOAD_TESTS = [
    {"config": CONFIGS[0], "depth": 2048, "ngl1": 0, "ngl2": 32},
    {"config": CONFIGS[0], "depth": 4096, "ngl1": 16, "ngl2": 32},
    {"config": CONFIGS[1], "depth": 2048, "ngl1": 0, "ngl2": 20},
    {"config": CONFIGS[2], "depth": 8192, "ngl1": 0, "ngl2": 80},
]


def measure_context_decay(config, depths):
    if not depths:
        return {"kv_bytes_per_token": 0.0, "throughputs": [], "decay_ratios": []}

    total_layers = config["total_layers"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    element_bytes = config["element_bytes"]
    model_bytes = float(config["model_bytes"])
    gpu_bw = float(config["gpu_bw_gbps"]) * 1e9

    kv_bytes_per_token = float(2 * total_layers * n_kv_heads * head_dim * element_bytes)

    throughputs = []
    for d in depths:
        total_bytes = model_bytes + float(d) * kv_bytes_per_token
        t_sec = total_bytes / gpu_bw
        throughputs.append(1.0 / t_sec)

    base_tp = throughputs[0]
    decay_ratios = [tp / base_tp for tp in throughputs]

    return {
        "kv_bytes_per_token": kv_bytes_per_token,
        "throughputs": throughputs,
        "decay_ratios": decay_ratios,
    }


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
