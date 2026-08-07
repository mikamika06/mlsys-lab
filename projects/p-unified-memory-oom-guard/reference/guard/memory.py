def measure_footprint(config):
    ctx = config.get("context_length", 512)
    layers = config.get("num_layers", 32)
    hidden = config.get("hidden_size", 4096)
    bytes_per_param = config.get("bytes_per_param", 2)
    base = layers * hidden * hidden * 4 * bytes_per_param // 1000000
    kv = ctx * layers * hidden * 2 * bytes_per_param // 1000000
    return base + kv + 128
