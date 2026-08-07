def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    m = {"zero_dropped_requests": 0.0, "error_rate_bounded": 0.0}
    try:
        mgr = ref.get_reference_manager()
        res = [mgr.predict("v1", i) for i in range(10)]
        if len(res) == 10 and all(x is not None for x in res):
            m["zero_dropped_requests"] = 1.0
            m["error_rate_bounded"] = 1.0
    except Exception:
        pass
    return m
