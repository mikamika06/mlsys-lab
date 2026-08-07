def aggregate_profile(data):
    total_time = 0.0
    total_invocations = 0
    for r in data.get("records", []):
        total_time += r.get("time_ms", 0.0) * r.get("invocations", 1)
        total_invocations += r.get("invocations", 1)
    return {
        "total_time": total_time,
        "total_invocations": total_invocations,
        "layer_count": len(data.get("records", []))
    }
