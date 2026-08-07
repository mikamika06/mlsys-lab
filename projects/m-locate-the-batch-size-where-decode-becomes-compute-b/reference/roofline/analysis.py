def find_decode_compute_bound_batch_size(model_config, hardware_specs):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    context_len = model_config.get("context_len", 1024)

    peak_flops = hardware_specs["peak_flops"]
    peak_bandwidth = hardware_specs["peak_bandwidth"]
    roofline_knee = peak_flops / peak_bandwidth

    weight_bytes_per_token = num_params * bytes_per_param
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param * context_len
    flops_per_token = 2 * num_params

    max_b = 16384
    for b in range(1, max_b + 1):
        total_bytes = weight_bytes_per_token + b * kv_bytes_per_token
        total_flops = b * flops_per_token
        intensity = total_flops / total_bytes
        if intensity >= roofline_knee:
            return b
    return max_b


def calculate_operational_intensity(model_config, batch_size, seq_len, phase="decode"):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]

    if phase == "prefill":
        flops = 2 * num_params * batch_size * seq_len
        bytes_transferred = (num_params * bytes_per_param) + (
            2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * seq_len
        )
    else:
        flops = 2 * num_params * batch_size
        bytes_transferred = (num_params * bytes_per_param) + (
            2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * seq_len
        )
    return flops / bytes_transferred
