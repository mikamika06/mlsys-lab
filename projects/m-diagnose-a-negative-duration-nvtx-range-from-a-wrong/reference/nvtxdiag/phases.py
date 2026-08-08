def rank_phases_by_self_time(spans):
    children_map = {}
    for s in spans:
        pid = s["parent_id"]
        if pid is not None:
            children_map.setdefault(pid, []).append(s)

    self_times = {}
    for s in spans:
        total_dur = s["end_ns"] - s["start_ns"]
        child_spans = children_map.get(s["id"], [])
        child_dur = sum(c["end_ns"] - c["start_ns"] for c in child_spans)
        self_dur = total_dur - child_dur
        name = s["name"]
        self_times[name] = self_times.get(name, 0) + self_dur

    ranked = sorted(self_times.items(), key=lambda x: (-x[1], x[0]))
    return [name for name, _ in ranked[:5]]
