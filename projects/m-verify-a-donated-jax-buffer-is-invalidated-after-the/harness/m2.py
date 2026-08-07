import ref

def check(workdir):
    out = {"savings_calculated": 0.0, "breakeven_matched": 0.0}
    try:
        from jaxserv.bench import compute_breakeven_requests, measure_peak_memory_savings
    except Exception as e:
        out["_note"] = f"Failed to import bench module: {type(e).__name__}: {e}"
        return out

    savings_ok = True
    for shape, dtype_b, num_u in ref.TEST_SHAPES:
        want = ref.measure_peak_memory_savings(shape, dtype_b, num_u)
        got = measure_peak_memory_savings(shape, dtype_b, num_u)
        if got != want:
            savings_ok = False
            out["_note"] = f"Memory savings mismatch: got {got}, expected {want}"
            break

    if savings_ok:
        out["savings_calculated"] = 1.0

    breakeven_ok = True
    for compile_ms, eager_ms, compiled_ms in ref.BREAKEVEN_CASES:
        want = ref.compute_breakeven_requests(compile_ms, eager_ms, compiled_ms)
        got = compute_breakeven_requests(compile_ms, eager_ms, compiled_ms)
        if got != want:
            breakeven_ok = False
            out["_note"] = f"Breakeven mismatch: got {got}, expected {want}"
            break

    if breakeven_ok:
        out["breakeven_matched"] = 1.0

    return out
