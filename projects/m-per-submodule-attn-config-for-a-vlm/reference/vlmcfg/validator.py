def validate_config(config):
    subs = config.get("submodules", [])
    for sub in subs:
        if "index" not in sub or "kind" not in sub:
            return False
        if sub.get("kv_heads", 0) <= 0 or sub.get("head_dim", 0) <= 0:
            return False
    return True
