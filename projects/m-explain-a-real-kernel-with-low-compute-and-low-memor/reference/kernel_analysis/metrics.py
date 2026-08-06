def compute_metrics(k):
    gflops = (k["flops"] / (k["duration_ms"] * 1e-3)) / 1e9
    intensity = k["flops"] / k["bytes_transferred"]
    achieved_bw = (k["bytes_transferred"] / (k["duration_ms"] * 1e-3)) / 1e9
    compute_pct = (gflops / k["peak_gflops"]) * 100.0
    memory_pct = (achieved_bw / k["peak_bandwidth_gbps"]) * 100.0
    gap_pct = 100.0 - compute_pct
    return {
        "gflops": gflops,
        "intensity": intensity,
        "compute_pct": compute_pct,
        "memory_pct": memory_pct,
        "gap_pct": gap_pct,
    }
