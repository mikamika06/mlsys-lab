import ref

def check(workdir):
    from flopcalc.mfu import calculate_mfu
    out = {"mfu_matched": 0.0}

    tokens_per_sec = 1000.0
    param_count = 10000000.0
    peak_flops = 1e12

    want = ref.compute_mfu(tokens_per_sec, param_count, peak_flops)
    try:
        got = calculate_mfu(tokens_per_sec, param_count, peak_flops)
    except Exception as e:
        out["_note"] = f"calculate_mfu raised {type(e).__name__}"
        return out

    if got is not None and abs(got - want) < 1e-5:
        out["mfu_matched"] = 1.0
    else:
        out["_note"] = f"got mfu {got}, expected {want}"
    return out
