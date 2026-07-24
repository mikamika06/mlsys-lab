def kv_cache_bytes(config, seq_len, batch_size):
    # BUG: uses num_attention_heads instead of num_kv_heads (blind to
    # GQA/MQA), and only counts one of K/V instead of both (missing *2).
    return (
        batch_size * seq_len * config["num_attention_heads"] * config["head_dim"]
        * config["num_layers"] * config["bytes_per_elem"]
    )
