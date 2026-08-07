from vlmattn.config import parse_submodule_configs


def compute_submodule_bytes(config, batch_size, seq_len):
    parsed = parse_submodule_configs(config)
    total_bytes = 0
    for name, cfg in parsed.items():
        kv_heads = cfg["kv_heads"]
        head_dim = cfg["head_dim"]
        effective_len = min(seq_len, cfg["window"]) if cfg["type"] == "sliding" else seq_len
        layer_bytes = 2 * batch_size * effective_len * kv_heads * head_dim * 2
        total_bytes += layer_bytes
    return total_bytes


def validate_submodule_constraints(config):
    parsed = parse_submodule_configs(config)
    for name, cfg in parsed.items():
        if cfg["kv_heads"] <= 0 or cfg["head_dim"] <= 0:
            return False
    return True
