import numpy as np

CONFIGS = [
    {"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000, "weight_bytes": 14 * 10**9},
    {"hidden_size": 8192, "num_layers": 80, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000, "weight_bytes": 70 * 10**9},
    {"hidden_size": 2048, "num_layers": 24, "num_kv_heads": 4, "head_dim": 128, "vocab_size": 32000, "weight_bytes": 3 * 10**9},
]

SWEEPS = [
    {
        "config_idx": 0,
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "measured_tokens_per_sec": [1200.0, 2300.0, 4400.0, 8200.0, 14000.0, 21000.0],
        "memory_bandwidth": 900.0 * 10**9,
        "compute_capacity": 300.0 * 10**12
    },
    {
        "config_idx": 1,
        "batch_sizes": [1, 2, 4, 8, 16],
        "measured_tokens_per_sec": [250.0, 490.0, 950.0, 1800.0, 3200.0],
        "memory_bandwidth": 900.0 * 10**9,
        "compute_capacity": 300.0 * 10**12
    }
]

def compute_decode_roofline(config, batch_size, memory_bandwidth, compute_capacity, context_length=1024):
    kv_bytes_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * 2
    total_bytes_per_token = config["weight_bytes"] + batch_size * context_length * kv_bytes_per_token
    flops_per_token = 2 * config["num_layers"] * (4 * config["hidden_size"] * config["hidden_size"] + 2 * config["hidden_size"] * context_length)
    time_mem = total_bytes_per_token / memory_bandwidth
    time_compute = flops_per_token / compute_capacity
    time_per_token = max(time_mem, time_compute)
    tokens_per_sec = batch_size / time_per_token
    return tokens_per_sec

def validate_sweep(sweep, max_rel_err=0.20):
    cfg = CONFIGS[sweep["config_idx"]]
    bw = sweep["memory_bandwidth"]
    cc = sweep["compute_capacity"]
    measured = sweep["measured_tokens_per_sec"]
    batch_sizes = sweep["batch_sizes"]
    rel_errs = []
    for bs, m_val in zip(batch_sizes, measured):
        pred = compute_decode_roofline(cfg, bs, bw, cc)
        err = abs(pred - m_val) / m_val
        rel_errs.append(err)
    max_err = float(np.max(rel_errs))
    return {"max_rel_err": max_err, "passed": max_err <= max_rel_err, "rel_errs": rel_errs}
