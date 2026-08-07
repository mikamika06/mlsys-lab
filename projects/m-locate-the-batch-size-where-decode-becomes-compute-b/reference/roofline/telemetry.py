def extract_achieved_hbm_bandwidth(measured_metrics, model_config):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]

    batch_size = measured_metrics["batch_size"]
    avg_seq_len = measured_metrics["avg_seq_len"]
    tokens_per_second = measured_metrics["tokens_per_second"]
    time_seconds = measured_metrics["time_seconds"]

    weight_bytes = num_params * bytes_per_param
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * avg_seq_len
    bytes_per_decode_step = weight_bytes + kv_bytes

    total_tokens = tokens_per_second * time_seconds
    num_decode_steps = total_tokens / batch_size
    total_bytes = bytes_per_decode_step * num_decode_steps

    achieved_bandwidth_gbps = (total_bytes / time_seconds) / 1e9
    return achieved_bandwidth_gbps
