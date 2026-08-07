def check(workdir):
    from audit.core import inspect_fusion
    m = {"fusion_matched": 0.0}
    try:
        res = inspect_fusion("test")
        if isinstance(res, dict) and res.get("fused") is True:
            m["fusion_matched"] = 1.0
    except Exception:
        pass
    return m
