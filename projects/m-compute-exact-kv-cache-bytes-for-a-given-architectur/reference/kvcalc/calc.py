def compute_kv_cache_bytes(config, num_ctx):
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    bytes_per_elem = config["bytes_per_elem"]
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem * num_ctx
