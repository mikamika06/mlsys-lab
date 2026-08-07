import ref

def check(workdir):
    m = {"cache_speedup_ok": 0.0}
    try:
        from trt_builder.cache import enable_timing_cache
        cfg = {}
        res = enable_timing_cache(cfg)
        if res.get("timing_cache_enabled") is True:
            m["cache_speedup_ok"] = 1.0
    except Exception:
        pass
    return m
