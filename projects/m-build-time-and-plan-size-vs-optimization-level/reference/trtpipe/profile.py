def analyze_build_tradeoffs(configs):
    """Analyze build time and plan size vs optimization level relative to baseline level 0."""
    results = []
    base_size = None
    for cfg in configs:
        if cfg["optimization_level"] == 0:
            base_size = float(cfg["plan_size_bytes"])
            break
    if base_size is None or base_size == 0:
        base_size = float(configs[0]["plan_size_bytes"]) if configs else 1.0

    for cfg in configs:
        level = cfg["optimization_level"]
        build_time = cfg["build_time_sec"]
        plan_size = cfg["plan_size_bytes"]
        ratio = float(plan_size) / base_size
        results.append({
            "optimization_level": level,
            "build_time_sec": build_time,
            "plan_size_bytes": plan_size,
            "size_ratio": ratio,
        })
    return results
