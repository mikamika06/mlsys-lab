def kv_cache_bytes(config, seq_len, batch_size):
    return (
        2 * batch_size * seq_len * config["num_kv_heads"] * config["head_dim"]
        * config["num_layers"] * config["bytes_per_elem"]
    )
