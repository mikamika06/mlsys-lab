def check(workdir):
    from shapes.verifier import Dim, resolve_module_signature

    out = {
        "auto_static_match": 0.0,
        "auto_varying_match": 0.0,
        "dynamic_match": 0.0,
        "explicit_match": 0.0,
    }

    try:
        dims = [Dim.auto("batch")]
        res = resolve_module_signature(dims, [(32,), (32,)])
        if res[0].min_val == 32 and res[0].max_val == 32:
            out["auto_static_match"] = 1.0
    except Exception:
        pass

    try:
        dims = [Dim.auto("seq")]
        resolve_module_signature(dims, [(128,), (256,)])
    except ValueError:
        out["auto_varying_match"] = 1.0
    except Exception:
        pass

    try:
        dims = [Dim.dynamic("seq")]
        res = resolve_module_signature(dims, [(128,), (512,)])
        if res[0].min_val == 128 and res[0].max_val == 512:
            out["dynamic_match"] = 1.0
    except Exception:
        pass

    try:
        dims = [Dim("batch", 1, 16)]
        res = resolve_module_signature(dims, [(8,), (16,)])
        if res[0].min_val == 1 and res[0].max_val == 16:
            out["explicit_match"] = 1.0
    except Exception:
        pass

    return out
