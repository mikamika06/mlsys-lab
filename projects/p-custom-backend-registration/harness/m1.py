def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from compiler.backend import BackendRegistry
    reg = BackendRegistry()
    m = {"registered_ok": 0.0}
    try:
        reg.register("test_backend", object)
        if reg.get("test_backend") is object:
            m["registered_ok"] = 1.0
    except Exception:
        pass
    return m
