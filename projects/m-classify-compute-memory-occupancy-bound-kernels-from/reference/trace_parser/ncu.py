def classify_ncu(metrics):
    res = {}
    for m in metrics:
        if m["sm_pct"] >= 80.0:
            res[m["name"]] = "compute_bound"
        elif m["mem_pct"] >= 80.0:
            res[m["name"]] = "memory_bound"
        elif m["warps_pct"] < 60.0:
            res[m["name"]] = "occupancy_bound"
        else:
            res[m["name"]] = "latency_bound"
    return res
