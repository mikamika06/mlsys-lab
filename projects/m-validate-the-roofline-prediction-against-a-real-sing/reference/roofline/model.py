import numpy as np

def compute_decode_roofline(config, batch_size, memory_bandwidth, compute_capacity, context_length=1024):
    kv_bytes_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * 2
    total_bytes_per_token = config["weight_bytes"] + batch_size * context_length * kv_bytes_per_token
    flops_per_token = 2 * config["num_layers"] * (4 * config["hidden_size"] * config["hidden_size"] + 2 * config["hidden_size"] * context_length)
    time_mem = total_bytes_per_token / memory_bandwidth
    time_compute = flops_per_token / compute_capacity
    time_per_token = max(time_mem, time_compute)
    tokens_per_sec = batch_size / time_per_token
    return tokens_per_sec
