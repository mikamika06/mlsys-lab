import ref

def check(workdir):
    from ort_tune.arena import configure_arena
    m = {"arena_config_ok": 0.0}
    cfg = configure_arena("default", 2048)
    if isinstance(cfg, dict) and cfg.get("enable_arena") is True and cfg.get("chunk_size") == 2048:
        m["arena_config_ok"] = 1.0
    return m
