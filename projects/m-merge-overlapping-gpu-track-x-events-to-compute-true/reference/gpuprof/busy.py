def compute_gpu_busy_time(x_events, stream_ids=None):
    intervals = []
    for ev in x_events:
        if ev.get("cat") != "gpu_op":
            continue
        if stream_ids is not None:
            st = ev.get("args", {}).get("stream")
            if st not in stream_ids:
                continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        if dur > 0:
            intervals.append((ts, ts + dur))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: x[0])
    merged = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append(cur_end - cur_start)
            cur_start, cur_end = start, end
    merged.append(cur_end - cur_start)
    return float(sum(merged))
