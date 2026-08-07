def check(workdir):
    from install.builder import lock_dependencies
    m = {"set_consistent": 0.0}
    try:
        locked = lock_dependencies()
    except Exception:
        return m
    if isinstance(locked, dict) and "torch" in locked and "cuda" in locked:
        m["set_consistent"] = 1.0
    return m
