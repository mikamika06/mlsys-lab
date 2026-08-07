def parse_key_averages(trace_json):
    events = trace_json.get("traceEvents", [])
    stats = {}
    for ev in events:
        if "dur" not in ev or "name" not in ev:
            continue
        name = ev["name"]
        dur = ev["dur"]
        if name not in stats:
            stats[name] = {"count": 0, "total_us": 0.0, "self_us": 0.0}
        stats[name]["count"] += 1
        stats[name]["total_us"] += float(dur)
        stats[name]["self_us"] += float(dur)

    result = []
    for name, data in stats.items():
        result.append({
            "name": name,
            "count": data["count"],
            "total_us": data["total_us"],
            "self_us": data["self_us"],
            "avg_us": data["total_us"] / data["count"] if data["count"] > 0 else 0.0
        })
    return sorted(result, key=lambda x: x["total_us"], reverse=True)


def compute_gpu_metrics(trace_json):
    events = trace_json.get("traceEvents", [])
    gpu_events = [ev for ev in events if ev.get("cat") == "kernel" and "dur" in ev]
    if not gpu_events:
        return {"busy_fraction": 0.0, "idle_gaps": []}

    intervals = []
    for ev in gpu_events:
        start = float(ev["ts"])
        dur = float(ev["dur"])
        intervals.append((start, start + dur))

    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total_active = sum(end - start for start, end in merged)
    min_ts = merged[0][0]
    max_ts = merged[-1][1]
    total_span = max_ts - min_ts if max_ts > min_ts else 1.0

    busy_fraction = total_active / total_span if total_span > 0 else 0.0

    idle_gaps = []
    for i in range(len(merged) - 1):
        gap = merged[i+1][0] - merged[i][1]
        if gap > 0:
            idle_gaps.append(gap)

    return {"busy_fraction": busy_fraction, "idle_gaps": idle_gaps}
