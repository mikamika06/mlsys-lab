import random


def get_test_configs():
    random.seed(42)
    configs = []
    for _ in range(5):
        num_layers = random.choice([24, 32, 40])
        num_kv_heads = random.choice([2, 4, 8])
        head_dim = random.choice([64, 128])
        bytes_per_elem = random.choice([2, 4])
        configs.append({
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "bytes_per_elem": bytes_per_elem
        })
    return configs


def compute_kv_bytes(config, num_ctx):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["bytes_per_elem"]
    return per_token * num_ctx


def find_max_ctx(config, weights_bytes, vram_bytes):
    if weights_bytes >= vram_bytes:
        return 0
    available = vram_bytes - weights_bytes
    per_token = compute_kv_bytes(config, 1)
    if per_token <= 0:
        return 0
    return int(available // per_token)
