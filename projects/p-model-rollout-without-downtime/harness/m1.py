def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    m = {"version_load_ok": 0.0, "parallel_active": 0.0}
    try:
        mgr = ref.get_reference_manager()
        if "v1" in mgr.versions and "v2" in mgr.versions:
            m["version_load_ok"] = 1.0
        if len(mgr.versions) == 2:
            m["parallel_active"] = 1.0
    except Exception:
        pass
    return m
