def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from guard.limiter import RuntimeLimiter

    m = {"limited_ok": 0.0}
    limiter = RuntimeLimiter(1500)
    res = limiter.check_and_apply(2000)
    if res in ["degrade", "block", False]:
        m["limited_ok"] = 1.0
    return m
