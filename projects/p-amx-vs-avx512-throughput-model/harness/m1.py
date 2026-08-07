def check(workdir):
    from amx_model import model
    m = {"amx_ok": 0.0, "avx512_ok": 0.0}
    try:
        v1 = model.predict_amx(64, 64, 64, "int8")
        if isinstance(v1, (int, float)) and v1 > 0:
            m["amx_ok"] = 1.0
    except Exception:
        pass
    try:
        v2 = model.predict_avx512(64, 64, 64, "int8")
        if isinstance(v2, (int, float)) and v2 > 0:
            m["avx512_ok"] = 1.0
    except Exception:
        pass
    return m
