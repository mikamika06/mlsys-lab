def analyze_bandwidth(benchmark_records):
    fused_times = [r["fused_time_ms"] for r in benchmark_records if r["kernel"] == "fused"]
    unfused_times = [r["unfused_time_ms"] for r in benchmark_records if r["kernel"] == "unfused"]
    if not fused_times or not unfused_times:
        return 0.0
    avg_fused = sum(fused_times) / len(fused_times)
    avg_unfused = sum(unfused_times) / len(unfused_times)
    if avg_fused <= 0:
        return 0.0
    return float(avg_unfused / avg_fused)
