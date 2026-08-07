def compute_kv_bytes(config, seq_len, batch_size, dtype_bytes):
    num_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    total_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    return batch_size * seq_len * total_bytes_per_token
