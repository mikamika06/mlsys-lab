def kv_bytes(config, seq_len, bytes_per_elem=2):
    l = config["num_layers"]
    kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    return 2 * l * seq_len * kv_heads * head_dim * bytes_per_elem
