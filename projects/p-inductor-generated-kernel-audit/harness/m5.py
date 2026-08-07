def check(workdir):
    from audit.core import optimize_both_sizes
    m = {"speedup_ok": 0.0}
    try:
        res = optimize_both_sizes(None, [512, 1024])
        if isinstance(res, dict) and all(res.values()):
            m["speedup_ok"] = 1.0
    except Exception:
        pass
    return m
