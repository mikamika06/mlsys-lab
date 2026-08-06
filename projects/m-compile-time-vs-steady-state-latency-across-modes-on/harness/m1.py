import ref

def check(workdir):
    from compengine.modes import select_mode

    out = {"modes_matched": 0.0}
    modes = ["default", "reduce-overhead", "max-autotune", "eager"]
    ok = 0
    for m in modes:
        try:
            res = select_mode(m)
            if m == "eager" and res is None:
                ok += 1
            elif isinstance(res, dict) and res.get("backend") == "inductor":
                ok += 1
        except Exception:
            pass
    out["modes_matched"] = float(ok)
    return out
