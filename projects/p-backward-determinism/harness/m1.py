def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from det.analyzer import locate_source
    m = {"api_ok": 0.0, "localized": 0.0}
    try:
        res = locate_source()
        m["api_ok"] = 1.0
        if isinstance(res, dict) and "atomics" in res:
            m["localized"] = 1.0
    except Exception:
        pass
    return m
