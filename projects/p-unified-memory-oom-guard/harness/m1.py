def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from guard.memory import measure_footprint

    m = {"measured_ok": 0.0}
    cfg = ref.get_sample_config()
    val = measure_footprint(cfg)
    if isinstance(val, (int, float)) and val > 0:
        m["measured_ok"] = 1.0
    return m
