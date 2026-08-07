CONFIGS = [
    {
        "default_attention": {"kv_heads": 8, "head_dim": 128, "window": 2048, "type": "full"},
        "submodules": {
            "vision": {"kv_heads": 16, "head_dim": 64, "type": "sliding", "window": 256},
            "language": {"kv_heads": 8, "head_dim": 128, "type": "full"}
        }
    },
    {
        "default_attention": {"kv_heads": 4, "head_dim": 64, "window": 1024, "type": "sliding"},
        "submodules": {
            "projector": {"kv_heads": 4, "head_dim": 64, "type": "full"},
            "decoder": {"kv_heads": 8, "head_dim": 64, "type": "sliding", "window": 512}
        }
    },
    {
        "default_attention": {"kv_heads": 2, "head_dim": 256, "window": 4096, "type": "full"},
        "submodules": {
            "encoder": {"kv_heads": 4, "head_dim": 128, "type": "full"}
        }
    }
]


def parse_submodule_configs(config):
    submodules = config.get("submodules", {})
    parsed = {}
    default_attn = config.get("default_attention", {"kv_heads": 8, "head_dim": 128, "window": 2048})
    for name, sub_cfg in submodules.items():
        merged = {**default_attn, **sub_cfg}
        parsed[name] = {
            "kv_heads": merged.get("kv_heads", 8),
            "head_dim": merged.get("head_dim", 128),
            "window": merged.get("window", 2048),
            "type": merged.get("type", "full")
        }
    return parsed


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
