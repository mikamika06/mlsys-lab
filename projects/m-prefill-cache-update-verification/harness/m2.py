def check(workdir):
    from cacheval.latency import analyze_latencies
    from cacheval.memory import compute_peak_memory_delta

    lat_res = analyze_latencies([15.0, 16.0, 14.0], [3.0, 3.2, 2.8])
    lat_ok = 1.0 if lat_res.get("valid") and lat_res.get("stateful_mean") < lat_res.get("stateless_mean") else 0.0

    mem_res = compute_peak_memory_delta(1024 * 1024, 1500 * 1024)
    mem_ok = 1.0 if mem_res.get("delta_bytes") == 1500 * 1024 - 1024 * 1024 else 0.0

    out = {
        "latency_checked": lat_ok,
        "memory_delta_checked": mem_ok
    }
    return out
