import ref

def check(workdir):
    m = {"threading_ok": 0.0}
    try:
        from cpuopt.runtime import configure_runtime
        res = configure_runtime(None, threads=4, latency_hint="latency")
        if isinstance(res, dict) and res.get("configured") and res.get("threads") == 4:
            m["threading_ok"] = 1.0
    except Exception:
        pass
    return m
