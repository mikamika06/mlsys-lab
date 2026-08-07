def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dl.loader import configure_pinning

    m = {"pinning_ok": 0.0}
    try:
        if configure_pinning(True, True) is True and configure_pinning(False, True) is False:
            m["pinning_ok"] = 1.0
    except Exception:
        pass
    return m
