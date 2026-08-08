def total_activation_memory(config, b, s, dtype_bytes):
    total = 0
    for layer in config["layers"]:
        h = layer["hidden_dim"]
        heads = layer["num_heads"]
        total += b * s * h * 2 * dtype_bytes + b * heads * s * s * dtype_bytes + b * s * (4 * h) * dtype_bytes
    return total
