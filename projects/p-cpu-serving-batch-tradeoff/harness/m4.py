def check(workdir):
    from serving import engine
    m = {"burst_ok": 0.0}
    try:
        lats = engine.simulate_burst(4, [5, 10], 4)
        if isinstance(lats, list) and len(lats) > 0:
            m["burst_ok"] = 1.0
    except Exception:
        pass
    return m
