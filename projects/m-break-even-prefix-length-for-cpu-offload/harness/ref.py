CONFIGS = [
    {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "num_params": 7e9},
    {"num_layers": 40, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "num_params": 13e9},
    {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "num_params": 70e9},
    {"num_layers": 16, "num_kv_heads": 4, "head_dim": 64, "dtype_bytes": 2, "num_params": 1.5e9},
    {"num_layers": 48, "num_kv_heads": 16, "head_dim": 128, "dtype_bytes": 2, "num_params": 32e9},
]

HARDWARE = [
    {"gpu_tflops": 150.0, "launch_overhead_s": 0.002},
    {"gpu_tflops": 312.0, "launch_overhead_s": 0.001},
    {"gpu_tflops": 80.0, "launch_overhead_s": 0.005},
    {"gpu_tflops": 500.0, "launch_overhead_s": 0.0005},
    {"gpu_tflops": 200.0, "launch_overhead_s": 0.002},
]

TIERS = [
    {
        "ram": {"bandwidth_gbps": 64.0, "latency_s": 0.0005},
        "nvme": {"bandwidth_gbps": 7.0, "latency_s": 0.005},
        "s3": {"bandwidth_gbps": 1.2, "latency_s": 0.08},
    },
    {
        "ram": {"bandwidth_gbps": 32.0, "latency_s": 0.001},
        "nvme": {"bandwidth_gbps": 3.5, "latency_s": 0.01},
        "s3": {"bandwidth_gbps": 0.8, "latency_s": 0.12},
    },
    {
        "ram": {"bandwidth_gbps": 128.0, "latency_s": 0.0002},
        "nvme": {"bandwidth_gbps": 14.0, "latency_s": 0.002},
        "s3": {"bandwidth_gbps": 2.5, "latency_s": 0.05},
    },
    {
        "ram": {"bandwidth_gbps": 48.0, "latency_s": 0.0008},
        "nvme": {"bandwidth_gbps": 5.0, "latency_s": 0.008},
        "s3": {"bandwidth_gbps": 1.0, "latency_s": 0.15},
    },
    {
        "ram": {"bandwidth_gbps": 96.0, "latency_s": 0.0003},
        "nvme": {"bandwidth_gbps": 10.0, "latency_s": 0.003},
        "s3": {"bandwidth_gbps": 1.8, "latency_s": 0.06},
    },
]


def compute_kv_bytes(config, prefix_len):
    return 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"] * prefix_len


def estimate_recompute_time(config, hw, prefix_len):
    flops = 2 * config["num_params"] * prefix_len
    gpu_flops = hw["gpu_tflops"] * 1e12
    return (flops / gpu_flops) + hw.get("launch_overhead_s", 0.0)


def estimate_load_time(config, tier, prefix_len):
    kv_bytes = compute_kv_bytes(config, prefix_len)
    bw_bps = tier["bandwidth_gbps"] * 1e9
    return (kv_bytes / bw_bps) + tier.get("latency_s", 0.0)


def compute_breakeven_prefix_length(config, hw, tier):
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


def select_optimal_tier(config, hw, tiers, prefix_len, max_latency_s):
    candidates = {
        "recompute": estimate_recompute_time(config, hw, prefix_len)
    }
    for name, tier in tiers.items():
        candidates[name] = estimate_load_time(config, tier, prefix_len)
    valid = {k: v for k, v in candidates.items() if v <= max_latency_s}
    if valid:
        best_name = min(valid, key=lambda k: (valid[k], k))
    else:
        best_name = min(candidates, key=lambda k: (candidates[k], k))
    return best_name


def evaluate_workload_latencies(config, hw, tiers, workload_prefix_lengths):
    result = {
        "recompute": [estimate_recompute_time(config, hw, L) for L in workload_prefix_lengths]
    }
    for name, tier in tiers.items():
        result[name] = [estimate_load_time(config, tier, L) for L in workload_prefix_lengths]
    return result
