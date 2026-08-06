from profile.memcpy import analyze_trace_memcpy


def compare_optimization_levels(traces_by_level):
    result = {}
    sorted_levels = sorted(traces_by_level.keys())
    if not sorted_levels:
        return result

    base_level = sorted_levels[0]
    base_analysis = analyze_trace_memcpy(traces_by_level[base_level])
    base_op_count = sum(s["count"] for s in base_analysis["node_stats"].values())
    base_dur = base_analysis["total_duration"]

    for lvl in sorted_levels:
        analysis = analyze_trace_memcpy(traces_by_level[lvl])
        op_count = sum(s["count"] for s in analysis["node_stats"].values())
        dur = analysis["total_duration"]

        op_reduction = (base_op_count - op_count) / base_op_count if base_op_count > 0 else 0.0
        speedup = (base_dur / dur) if dur > 0 else 1.0

        result[lvl] = {
            "total_op_count": op_count,
            "total_duration": dur,
            "memcpy_duration": analysis["memcpy_duration"],
            "op_reduction_ratio": op_reduction,
            "speedup_vs_baseline": speedup,
            "memcpy_ratio": analysis["memcpy_ratio"],
        }
    return result
