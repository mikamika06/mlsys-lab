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


def get_submodule_config(config, submodule_name):
    parsed = parse_submodule_configs(config)
    if submodule_name in parsed:
        return parsed[submodule_name]
    return {
        "kv_heads": config.get("default_attention", {}).get("kv_heads", 8),
        "head_dim": config.get("default_attention", {}).get("head_dim", 128),
        "window": config.get("default_attention", {}).get("window", 2048),
        "type": config.get("default_attention", {}).get("type", "full")
    }
