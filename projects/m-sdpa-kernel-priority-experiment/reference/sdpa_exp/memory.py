def compute_memory_blowup(config):
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["num_heads"]
    dt = config["dtype_bytes"]
    flash_mem = b * h * s * dt * 2
    math_mem = flash_mem + (b * h * s * s * dt)
    return float(math_mem / max(1, flash_mem))
