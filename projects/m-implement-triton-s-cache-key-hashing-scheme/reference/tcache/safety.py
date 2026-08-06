def is_safe_config(cfg):
    if cfg.get("stochastic", False):
        return False
    if cfg.get("dynamic_state", False):
        return False
    return True
