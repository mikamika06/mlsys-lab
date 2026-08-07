def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dl.loader import optimize_hotpath

    m = {"hotpath_clean": 0.0}
    try:
        res = optimize_hotpath(lambda x: x + 1, 10)
        if res == 11:
            m["hotpath_clean"] = 1.0
    except Exception:
        pass
    return m
