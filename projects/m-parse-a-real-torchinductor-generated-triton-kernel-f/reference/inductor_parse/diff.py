def diff_configs(default_config: dict, autotune_config: dict) -> dict:
    """Compare default and max-autotune Triton kernel configurations."""
    all_keys = set(default_config.keys()) | set(autotune_config.keys())
    changed = {}
    same = {}

    for k in sorted(all_keys):
        def_val = default_config.get(k)
        auto_val = autotune_config.get(k)
        if def_val != auto_val:
            changed[k] = {"default": def_val, "autotune": auto_val}
        else:
            same[k] = def_val

    return {"changed": changed, "same": same}
