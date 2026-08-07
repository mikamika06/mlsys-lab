def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    m = {"rollback_triggered": 0.0, "fallback_version_active": 0.0}
    try:
        pol = ref.get_reference_policy()
        should = pol.should_rollback(0.10)
        mgr = ref.get_reference_manager()
        mgr.rollback("v1")
        if should:
            m["rollback_triggered"] = 1.0
        if mgr.active_version == "v1":
            m["fallback_version_active"] = 1.0
    except Exception:
        pass
    return m
