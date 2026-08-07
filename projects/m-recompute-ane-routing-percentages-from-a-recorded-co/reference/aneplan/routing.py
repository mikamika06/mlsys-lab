def recompute_routing(profiler_export):
    ops = profiler_export.get("operations", [])
    total_ops = len(ops)
    if total_ops == 0:
        return {"ane_percentage": 0.0, "cpu_percentage": 0.0, "gpu_percentage": 0.0}
    counts = {"ANE": 0, "CPU": 0, "GPU": 0}
    for op in ops:
        target = op.get("compute_unit", "CPU")
        counts[target] = counts.get(target, 0) + 1
    return {
        "ane_percentage": round(counts["ANE"] * 100.0 / total_ops, 2),
        "cpu_percentage": round(counts["CPU"] * 100.0 / total_ops, 2),
        "gpu_percentage": round(counts["GPU"] * 100.0 / total_ops, 2),
    }
