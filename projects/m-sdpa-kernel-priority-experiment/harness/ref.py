import numpy as np

CONFIGS = [
    {"batch_size": 2, "seq_len": 1024, "num_heads": 8, "head_dim": 64, "dtype_bytes": 2},
    {"batch_size": 4, "seq_len": 2048, "num_heads": 16, "head_dim": 128, "dtype_bytes": 2},
    {"batch_size": 1, "seq_len": 4096, "num_heads": 32, "head_dim": 128, "dtype_bytes": 4},
]

def get_kernel_priority(config):
    if config["seq_len"] <= 2048 and config["head_dim"] <= 128:
        return ["flash", "math"]
    return ["math", "flash"]

def compute_memory_blowup(config):
    b = config["batch_size"]
    s = config["seq_len"]
    h = config["num_heads"]
    dt = config["dtype_bytes"]
    flash_mem = b * h * s * dt * 2
    math_mem = flash_mem + (b * h * s * s * dt)
    return float(math_mem / max(1, flash_mem))

def can_use_flash_attention(config):
    if config["head_dim"] not in (32, 64, 128, 256):
        return False
    if config["dtype_bytes"] == 4 and config["seq_len"] > 2048:
        return False
    return True
