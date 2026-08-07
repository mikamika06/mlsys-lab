def compute_matmul_flops(trace_data):
    events = trace_data.get("traceEvents", [])
    results = {}
    for ev in events:
        name = ev.get("name", "")
        args = ev.get("args", {})
        if "gemm" in name.lower() or "matmul" in name.lower():
            flops = args.get("flops", 0)
            dur = ev.get("dur", 1)
            achieved = (flops / (dur * 1e-6)) if dur > 0 else 0.0
            results[name] = achieved
    return results
