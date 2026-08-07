def adapter_size_bytes(config):
    total = 0
    dim = config["hidden_dim"]
    r = config["rank"]
    b = config["dtype_bytes"]
    for _ in config["modules"]:
        total += (dim * r * b) + (r * dim * b)
    return int(total)
