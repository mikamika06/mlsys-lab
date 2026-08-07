def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from guard.limiter import degrade_gracefully

    m = {"degraded_ok": 0.0}
    cfg = ref.get_sample_config()
    deg = degrade_gracefully(cfg, 1000)
    if isinstance(deg, dict) and deg.get("context_length", 999999) < cfg["context_length"]:
        m["degraded_ok"] = 1.0
    return m
