MODELS = [
    {"layers": 32, "hidden_dim": 4096, "kv_heads": 8, "head_dim": 128},
    {"layers": 48, "hidden_dim": 8192, "kv_heads": 8, "head_dim": 128},
    {"layers": 24, "hidden_dim": 2048, "kv_heads": 4, "head_dim": 128}
]

SYSTEMS = [
    {"swap_bandwidth_gbps": 32.0, "flops_per_token": 1.5e9, "block_size": 16},
    {"swap_bandwidth_gbps": 16.0, "flops_per_token": 8.0e8, "block_size": 32},
    {"swap_bandwidth_gbps": 64.0, "flops_per_token": 2.0e9, "block_size": 16}
]

LOGS = [
    [
        {"tokens_recomputed": 512, "block_count": 4, "mode": "recompute"},
        {"tokens_recomputed": 1024, "block_count": 8, "mode": "swap"}
    ],
    [
        {"tokens_recomputed": 256, "block_count": 2, "mode": "swap"},
        {"tokens_recomputed": 2048, "block_count": 16, "mode": "recompute"}
    ]
]

def compute_swap_cost(block_count, model_config, system_config):
    bytes_per_block = block_count * system_config["block_size"] * model_config["kv_heads"] * model_config["head_dim"] * 2 * model_config["layers"]
    bandwidth = system_config["swap_bandwidth_gbps"] * 1e9 / 8
    return bytes_per_block / bandwidth

def compute_recompute_cost(tokens, model_config, system_config):
    total_flops = tokens * system_config["flops_per_token"]
    peak_flops = 3e14
    return total_flops / peak_flops

def find_crossover(model_config, system_config):
    low = 1
    high = 20000
    best = high
    for b in range(low, high):
        sc = compute_swap_cost(b, model_config, system_config)
        tokens = b * system_config["block_size"]
        rc = compute_recompute_cost(tokens, model_config, system_config)
        if rc >= sc:
            best = b
            break
    return best

def compute_wasted_compute(preemption_log, model_config, system_config):
    total_wasted = 0.0
    for entry in preemption_log:
        tokens = entry["tokens_recomputed"]
        total_wasted += tokens * system_config["flops_per_token"]
    return total_wasted

def choose_cheaper_mode(context_len, model_config, system_config):
    blocks = (context_len + system_config["block_size"] - 1) // system_config["block_size"]
    sc = compute_swap_cost(blocks, model_config, system_config)
    rc = compute_recompute_cost(context_len, model_config, system_config)
    return "swap" if sc < rc else "recompute"
