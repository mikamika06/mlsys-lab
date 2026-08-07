def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    m = {"warmup_executed": 0.0, "cold_start_latency_avoided": 0.0}
    try:
        mgr = ref.get_reference_manager()
        mgr.warmup("v2", [1, 2, 3])
        if mgr.versions["v2"]["warmed"]:
            m["warmup_executed"] = 1.0
            m["cold_start_latency_avoided"] = 1.0
    except Exception:
        pass
    return m
