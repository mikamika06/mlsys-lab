def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from quant.budget import compute_effective_bpw

    m = {"bpw_calculated_ok": 0.0, "overhead_accounted": 0.0}
    val = compute_effective_bpw("fp4", 32, 8)
    if abs(val - 4.25) < 1e-5:
        m["bpw_calculated_ok"] = 1.0

    val2 = compute_effective_bpw("fp8", 64, 8)
    if abs(val2 - 8.125) < 1e-5:
        m["overhead_accounted"] = 1.0
    return m
