from memrunner.predictor import calculate_kv_cache_bytes, calculate_weight_bytes


def predict_decode_tok_s(config, seq_len, memory_bandwidth_gbps, batch_size=1):
    w_bytes = calculate_weight_bytes(config)
    kv_bytes = calculate_kv_cache_bytes(config, seq_len, batch_size)
    bytes_per_step = w_bytes + kv_bytes

    bandwidth_bytes_per_sec = memory_bandwidth_gbps * 1e9
    steps_per_sec = bandwidth_bytes_per_sec / bytes_per_step
    return steps_per_sec * batch_size
