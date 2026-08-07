def plan_bytes(config, seq_len, dtype_size, batch_size):
    total = 0
    for sub in config["submodules"]:
        kv_heads = sub["kv_heads"]
        head_dim = sub["head_dim"]
        total += batch_size * seq_len * kv_heads * head_dim * dtype_size * 2
    return total


def uniform_bytes(config, seq_len, dtype_size, batch_size):
    if not config["submodules"]:
        return 0
    max_kv = max(s["kv_heads"] for s in config["submodules"])
    max_hd = max(s["head_dim"] for s in config["submodules"])
    return len(config["submodules"]) * batch_size * seq_len * max_kv * max_hd * dtype_size * 2


def free_schedule(seq_len, dtype_size, step_count):
    res = []
    step_size = max(1, seq_len // step_count)
    current = 0
    for i in range(step_count):
        current = min(seq_len, current + step_size)
        res.append(current * dtype_size * 16)
    return res
