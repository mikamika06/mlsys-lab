def find_cache_break(base_cfg, new_cfg):
    for k, v in base_cfg.items():
        if new_cfg.get(k) != v:
            return k
    return None
