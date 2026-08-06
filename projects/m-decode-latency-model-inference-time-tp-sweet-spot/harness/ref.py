import numpy as np

CONFIGS = [
    {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128},
    {"hidden_size": 8192, "num_layers": 32, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128},
    {"hidden_size": 2048, "num_layers": 24, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 64}
]

HW_PROFILES = [
    {"name": "a100", "memory_bw_gbps": 1555.0, "compute_tflops": 312.0, "comm_bw_gbps": 300.0},
    {"name": "h100", "memory_bw_gbps": 3350.0, "compute_tflops": 1000.0, "comm_bw_gbps": 900.0}
]

def estimate_decode_latency(config, hw, tp_degree, batch_size):
    hidden = config["hidden_size"]
    layers = config["num_layers"]
    weights_bytes = hidden * hidden * 4 * layers * 2 / tp_degree
    comm_bytes = hidden * 4 * 2 * (tp_degree - 1) / tp_degree * batch_size
    mem_time = weights_bytes / (hw["memory_bw_gbps"] * 1e9)
    comm_time = comm_bytes / (hw["comm_bw_gbps"] * 1e9)
    return float((mem_time + comm_time) * layers * batch_size)

def shard_kv_heads(num_kv_heads, tp_degree):
    base = num_kv_heads // tp_degree
    rem = num_kv_heads % tp_degree
    shards = []
    for i in range(tp_degree):
        shards.append(base + (1 if i < rem else 0))
    return shards

def find_tp_sweet_spot(config, hw, batch_size, max_tp=8):
    best_tp = 1
    best_lat = float("inf")
    for tp in [1, 2, 4, 8]:
        if tp > max_tp:
            continue
        lat = estimate_decode_latency(config, hw, tp, batch_size)
        if lat < best_lat:
            best_lat = lat
            best_tp = tp
    return best_tp
