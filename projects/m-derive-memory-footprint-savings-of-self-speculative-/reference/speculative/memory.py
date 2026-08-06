def compute_weight_memory(config):
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]
    bytes_per_param = config.get("bytes_per_param", 2)
    attn_weights = 4 * (hidden_size * hidden_size)
    mlp_weights = 3 * (hidden_size * intermediate_size)
    layer_weights = attn_weights + mlp_weights
    total = (num_layers * layer_weights) + (vocab_size * hidden_size)
    return total * bytes_per_param


def compute_kv_cache_memory(config, batch_size, seq_len):
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    bytes_per_elem = config.get("bytes_per_elem", 2)
    cache_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    return batch_size * seq_len * cache_per_token
