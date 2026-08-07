def check(workdir):
    from serving import engine
    m = {"curve_ok": 0.0}
    try:
        res = engine.latency_curve([1, 4, 8, 16], 10.0, 2.0, 4)
        if len(res) == 4 and res[0] > 0 and res[3] > res[0]:
            m["curve_ok"] = 1.0
    except Exception:
        pass
    return m
