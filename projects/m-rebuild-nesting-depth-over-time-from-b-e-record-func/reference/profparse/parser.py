def parse_events(events):
    parsed = []
    stack = {}
    sorted_events = sorted(events, key=lambda x: x.get("ts", 0))
    for ev in sorted_events:
        ph = ev.get("ph")
        name = ev.get("name")
        ts = ev.get("ts", 0)
        pid = ev.get("pid", 0)
        tid = ev.get("tid", 0)
        key = (pid, tid, name)
        if ph == "B":
            stack.setdefault(key, []).append(ts)
        elif ph == "E":
            if key in stack and stack[key]:
                start_ts = stack[key].pop()
                dur = ts - start_ts
                parsed.append({"name": name, "pid": pid, "tid": tid, "ts": start_ts, "dur": dur, "ph": "X"})
            else:
                parsed.append({"name": name, "pid": pid, "tid": tid, "ts": ts, "dur": 0, "ph": ph})
        elif ph == "X":
            parsed.append({"name": name, "pid": pid, "tid": tid, "ts": ts, "dur": ev.get("dur", 0), "ph": "X"})
        else:
            parsed.append(ev)
    return parsed


def map_tracks(events, metadata):
    pid_map = metadata.get("pid_names", {})
    tid_map = metadata.get("tid_names", {})
    mapped = []
    for ev in events:
        new_ev = dict(ev)
        pid = ev.get("pid")
        tid = ev.get("tid")
        if pid in pid_map:
            new_ev["process_name"] = pid_map[pid]
        if tid in tid_map:
            new_ev["thread_name"] = tid_map[tid]
        mapped.append(new_ev)
    return mapped
