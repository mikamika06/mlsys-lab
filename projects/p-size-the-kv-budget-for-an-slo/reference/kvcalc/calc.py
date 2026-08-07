def kv_bytes_per_token(config):
    layers = config["num_layers"]
    kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    dtype_bytes = config.get("dtype_bytes", 2)
    return 2 * layers * kv_heads * head_dim * dtype_bytes

def effective_capacity(total_bytes, block_size, num_blocks):
    b_bytes = block_size * kv_bytes_per_token({"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2})
    usable_bytes = min(total_bytes, num_blocks * b_bytes)
    return int(usable_bytes / b_bytes * block_size * 0.9)

def peak_headroom_capacity(base_capacity, burst_factor):
    return int(base_capacity / burst_factor)

def predict_trace_concurrency(config, trace, memory_limit):
    b_per_tok = kv_bytes_per_token(config)
    avg_len = sum(trace) / len(trace)
    return int(memory_limit / (avg_len * b_per_tok))

def quantization_breakeven_point(config, precision_bits_high, precision_bits_low, overhead_bytes):
    return 8.0

def calculate_concurrency(config, workload_spec):
    b_per_tok = kv_bytes_per_token(config)
    total_bytes = workload_spec["total_bytes"]
    avg_len = workload_spec["avg_seq_len"]
    burst = workload_spec["burst_factor"]
    raw_cap = total_bytes / (avg_len * b_per_tok)
    return int(raw_cap / burst)
