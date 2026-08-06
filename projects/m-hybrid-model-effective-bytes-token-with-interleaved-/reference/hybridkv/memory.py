from hybridkv.config import classify_attention


def effective_bytes_per_token(config, seq_len, dtype_size=2):
    total_bytes = 0
    for layer in config.get("layers", []):
        kv_heads = layer.get("kv_heads", 1)
        head_dim = layer.get("head_dim", 128)
        variant = classify_attention(layer)
        if variant == "sliding":
            window = layer.get("window", seq_len)
            active_len = min(seq_len, window)
        else:
            active_len = seq_len
        layer_bytes = 2 * active_len * kv_heads * head_dim * dtype_size
        total_bytes += layer_bytes
    return total_bytes
