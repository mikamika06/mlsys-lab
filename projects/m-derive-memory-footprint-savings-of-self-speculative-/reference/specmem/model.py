def compute_parameter_bytes(config):
    bits_per_param = config.get("bits_per_param", 16)
    bytes_per_param = bits_per_param / 8.0
    total_params = config.get("total_params", 0)
    return int(total_params * bytes_per_param)


def compute_kv_cache_bytes(config, batch_size, seq_len):
    num_layers = config.get("num_layers", 32)
    num_kv_heads = config.get("num_kv_heads", 8)
    head_dim = config.get("head_dim", 128)
    bytes_per_elem = config.get("bytes_per_elem", 2)
    return int(2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * bytes_per_elem)
