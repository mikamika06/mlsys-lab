def check(workdir):
    from megacache.timing import measure_compile_time
    out = {"speedup_valid": 0.0}
    try:
        cold = measure_compile_time(False)
        warm = measure_compile_time(True)
        if warm < cold:
            out["speedup_valid"] = 1.0
        else:
            out["_note"] = f"warm time {warm} not less than cold time {cold}"
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {e}"
    return out
