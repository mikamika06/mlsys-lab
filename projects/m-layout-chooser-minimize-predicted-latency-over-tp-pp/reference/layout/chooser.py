CANDIDATE_LAYOUTS = [
    (1, 1, 8),
    (2, 1, 4),
    (4, 1, 2),
    (8, 1, 1),
    (1, 2, 4),
    (2, 2, 2),
    (4, 2, 1),
    (1, 4, 2),
    (2, 4, 1),
    (1, 8, 1),
]


def check_memory_fit(config: dict, tp: int, pp: int, vram_gb: float) -> bool:
    """Check if model weights, KV cache, and activation memory fit in per-GPU VRAM."""
    num_layers = config["num_layers"]
    weight_bytes = config["weight_bytes"]
    kv_per_token_layer = config["kv_bytes_per_token_per_layer"]
    act_per_token_layer = config["activation_bytes_per_token_per_layer"]
    max_seq_len = config["max_seq_len"]
    batch_size = config["batch_size"]

    layers_per_gpu = num_layers / pp
    weight_per_gpu = (weight_bytes / tp) / pp
    kv_per_gpu = (kv_per_token_layer * max_seq_len * batch_size * layers_per_gpu) / tp
    act_per_gpu = act_per_token_layer * max_seq_len * batch_size * layers_per_gpu

    total_bytes = weight_per_gpu + kv_per_gpu + act_per_gpu
    total_gb = total_bytes / (1024 ** 3)
    return total_gb <= vram_gb


def select_layout(config: dict, vram_gb: float, latency_table: dict) -> int:
    """Find index of valid 8-GPU layout minimizing predicted latency."""
    best_idx = -1
    best_latency = float("inf")

    for idx, (tp, pp, dp) in enumerate(CANDIDATE_LAYOUTS):
        if check_memory_fit(config, tp, pp, vram_gb):
            lat = latency_table.get((tp, pp, dp), float("inf"))
            if lat < best_latency:
                best_latency = lat
                best_idx = idx

    return best_idx
