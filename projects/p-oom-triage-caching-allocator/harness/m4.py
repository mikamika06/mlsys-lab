def check(workdir):
    m = {"config_ok": 0.0}
    try:
        from oom_triage.config import get_allocator_config
        cfg = get_allocator_config()
        if isinstance(cfg, int) and 0 < cfg <= 100:
            m["config_ok"] = 1.0
    except Exception:
        pass
    return m
