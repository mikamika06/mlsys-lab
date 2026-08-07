def check(workdir):
    from serving import engine
    m = {"max_throughput_ok": 0.0}
    try:
        opt = engine.max_throughput_point([1, 2, 4, 8, 16], 40.0, 4)
        if opt > 0:
            m["max_throughput_ok"] = 1.0
    except Exception:
        pass
    return m
