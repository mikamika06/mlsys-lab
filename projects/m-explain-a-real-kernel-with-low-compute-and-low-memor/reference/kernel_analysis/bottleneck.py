from kernel_analysis.metrics import compute_metrics


def classify_bottleneck(k):
    m = compute_metrics(k)
    occupancy_ratio = k["active_warps"] / k["max_warps"]
    if m["compute_pct"] < 25.0 and m["memory_pct"] < 25.0 and occupancy_ratio < 0.3:
        return "latency-bound-low-occupancy"
    elif m["compute_pct"] >= m["memory_pct"]:
        return "compute-bound"
    else:
        return "memory-bound"
