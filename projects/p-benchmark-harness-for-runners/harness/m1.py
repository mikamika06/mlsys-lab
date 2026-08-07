def check(workdir):
    from runner.core import measure_latency

    m = {"methodology_ok": 0.0}
    try:
        res = measure_latency([100, 50, 48, 52, 49, 51], warmup_count=2)
        if isinstance(res, dict) and "median" in res and "iqr" in res:
            if abs(res["median"] - 50.0) < 1.0:
                m["methodology_ok"] = 1.0
    except Exception:
        pass
    return m
