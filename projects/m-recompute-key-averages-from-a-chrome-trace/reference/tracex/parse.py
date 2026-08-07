def compute_key_averages(trace_data):
    events = trace_data.get("traceEvents", trace_data)
    durations = {}
    counts = {}
    self_times = {}
    complete_events = [e for e in events if e.get("ph") == "X" and "dur" in e]
    for ev in complete_events:
        name = ev.get("name", "unknown")
        dur = float(ev.get("dur", 0))
        counts[name] = counts.get(name, 0) + 1
        durations[name] = durations.get(name, 0.0) + dur
        children_dur = 0.0
        pid = ev.get("pid")
        tid = ev.get("tid")
        ts = float(ev.get("ts", 0))
        end = ts + dur
        for child in complete_events:
            if child.get("pid") == pid and child.get("tid") == tid:
                c_ts = float(child.get("ts", 0))
                c_dur = float(child.get("dur", 0))
                if c_ts >= ts and (c_ts + c_dur) <= end and (c_ts > ts or c_dur < dur):
                    children_dur += c_dur
        st = max(0.0, dur - children_dur)
        self_times[name] = self_times.get(name, 0.0) + st
    result = {}
    for name, total_dur in durations.items():
        cnt = counts[name]
        avg_dur = total_dur / cnt if cnt > 0 else 0.0
        st = self_times.get(name, 0.0)
        result[name] = {
            "count": cnt,
            "total_duration": total_dur,
            "average_duration": avg_dur,
            "total_self_time": st,
            "average_self_time": st / cnt if cnt > 0 else 0.0
        }
    return result
