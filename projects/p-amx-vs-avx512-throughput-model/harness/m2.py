def check(workdir):
    from amx_model import model
    m = {"shape_dep_ok": 0.0}
    try:
        r1 = model.analyze_shape(16, 16, 64, "int8")
        r2 = model.analyze_shape(64, 64, 64, "int8")
        if isinstance(r1, dict) and isinstance(r2, dict) and r1 != r2:
            m["shape_dep_ok"] = 1.0
    except Exception:
        pass
    return m
