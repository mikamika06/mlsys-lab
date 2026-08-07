def compute_gpu_metrics(trace_data):
    events = trace_data.get("traceEvents", trace_data)
    gpu_events = [e for e in events if "gpu" in str(e.get("cat", "")).lower() or "cuda" in str(e.get("name", "")).lower() or e.get("ph") == "X"]
    if not gpu_events:
        return {"busy_fraction": 0.0, "idle_gaps": 0.0}
    intervals = []
    for e in gpu_events:
        if e.get("ph") == "X" and "dur" in e:
            start = float(e.get("ts", 0))
            dur = float(e.get("dur", 0))
            intervals.append((start, start + dur))
    if not intervals:
        return {"busy_fraction": 0.0, "idle_gaps": 0.0}
    intervals.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= curr_end:
            curr_end = max(curr_end, end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = start, end
    merged.append((curr_start, curr_end))
    total_span = merged[-1][1] - merged[0][0]
    if total_span <= 0:
        return {"busy_fraction": 1.0, "idle_gaps": 0.0}
    busy_time = sum(end - start for start, end in merged)
    idle_time = total_span - busy_time
    busy_fraction = busy_time / total_span
    return {"busy_fraction": float(busy_fraction), "idle_gaps": float(idle_time)}
