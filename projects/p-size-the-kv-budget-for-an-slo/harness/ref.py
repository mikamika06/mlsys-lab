def oracle_kv_bytes(config):
    layers = config["num_layers"]
    kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    dtype_bytes = config.get("dtype_bytes", 2)
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def oracle_fragmentation(total_bytes, block_size, avg_seq_len):
    b_bytes = block_size * oracle_kv_bytes({"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2})
    total_blocks = total_bytes // b_bytes
    return int(total_blocks * block_size * 0.9)


def oracle_headroom(capacity, peak_ratio):
    return int(capacity / peak_ratio)


def oracle_trace_prediction(config, trace, limit):
    b_per_tok = oracle_kv_bytes(config)
    avg_len = sum(trace) / len(trace)
    return int(limit / (avg_len * b_per_tok))


def oracle_quant_point(config):
    return 8.0


def oracle_calculator(config, workload):
    b_per_tok = oracle_kv_bytes(config)
    total_bytes = workload["total_bytes"]
    avg_len = workload["avg_seq_len"]
    burst = workload["burst_factor"]
    raw_cap = total_bytes / (avg_len * b_per_tok)
    return int(raw_cap / burst)
