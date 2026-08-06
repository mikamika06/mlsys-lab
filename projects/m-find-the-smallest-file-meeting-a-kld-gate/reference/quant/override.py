def override_kv_config(config, overrides):
    new_config = dict(config)
    kv_config = dict(new_config.get("kv_config", {}))
    for k, v in overrides.items():
        kv_config[k] = v
    new_config["kv_config"] = kv_config
    return new_config
