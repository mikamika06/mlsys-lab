def resolve_args(defaults, yaml_cfg, env_cfg, cli_cfg):
    resolved = dict(defaults)
    for k, v in yaml_cfg.items():
        if v is not None:
            resolved[k] = v
    for k, v in env_cfg.items():
        if v is not None:
            resolved[k] = v
    for k, v in cli_cfg.items():
        if v is not None:
            resolved[k] = v
    return resolved
