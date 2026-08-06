import numpy as np

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

CONFIGS = [
    {
        "num_layers": 32,
        "weight_bytes": 16 * (1024 ** 3),
        "kv_bytes_per_token_per_layer": 2048,
        "activation_bytes_per_token_per_layer": 4096,
        "max_seq_len": 2048,
        "batch_size": 8,
    },
    {
        "num_layers": 64,
        "weight_bytes": 32 * (1024 ** 3),
        "kv_bytes_per_token_per_layer": 4096,
        "activation_bytes_per_token_per_layer": 8192,
        "max_seq_len": 4096,
        "batch_size": 4,
    },
    {
        "num_layers": 16,
        "weight_bytes": 8 * (1024 ** 3),
        "kv_bytes_per_token_per_layer": 1024,
        "activation_bytes_per_token_per_layer": 2048,
        "max_seq_len": 1024,
        "batch_size": 16,
    },
]

VRAM_TESTS = [8.0, 16.0, 24.0, 40.0, 80.0]

HISTOGRAMS = [
    np.array([100, 100, 100, 100]),
    np.array([400, 100, 50, 50]),
    np.array([1000, 0, 0, 0]),
    np.array([0, 0, 0, 0]),
]

LATENCY_TABLES = [
    {
        (1, 1, 8): 100.0, (2, 1, 4): 80.0, (4, 1, 2): 60.0, (8, 1, 1): 40.0,
        (1, 2, 4): 90.0, (2, 2, 2): 70.0, (4, 2, 1): 50.0,
        (1, 4, 2): 85.0, (2, 4, 1): 65.0, (1, 8, 1): 95.0,
    },
    {
        (1, 1, 8): 20.0, (2, 1, 4): 35.0, (4, 1, 2): 50.0, (8, 1, 1): 70.0,
        (1, 2, 4): 25.0, (2, 2, 2): 40.0, (4, 2, 1): 55.0,
        (1, 4, 2): 30.0, (2, 4, 1): 45.0, (1, 8, 1): 60.0,
    },
]


def check_memory_fit(config: dict, tp: int, pp: int, vram_gb: float) -> bool:
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


def straggler_factor(routing_histogram: np.ndarray) -> float:
    arr = np.asarray(routing_histogram, dtype=np.float64)
    if arr.size == 0 or np.sum(arr) == 0:
        return 1.0
    mean_val = np.mean(arr)
    if mean_val == 0:
        return 1.0
    return float(np.max(arr) / mean_val)


def select_layout(config: dict, vram_gb: float, latency_table: dict) -> int:
    best_idx = -1
    best_latency = float("inf")

    for idx, (tp, pp, dp) in enumerate(CANDIDATE_LAYOUTS):
        if check_memory_fit(config, tp, pp, vram_gb):
            lat = latency_table.get((tp, pp, dp), float("inf"))
            if lat < best_latency:
                best_latency = lat
                best_idx = idx

    return best_idx
