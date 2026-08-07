def check(workdir):
    from audit.core import apply_compilation_controls
    m = {"control_applied": 0.0}
    try:
        res = apply_compilation_controls({"test": 123})
        if isinstance(res, dict) and res.get("test") == 123:
            m["control_applied"] = 1.0
    except Exception:
        pass
    return m
