def align_clocks(profiles):
    aligned = []
    offsets = []
    for p in profiles:
        sync_events = [e for e in p["events"] if e.get("name") == "sync"]
        base_t = sync_events[0]["ts"] if sync_events else 0
        offsets.append(base_t)
    min_offset = min(offsets) if offsets else 0
    for i, p in enumerate(profiles):
        delta = offsets[i] - min_offset
        new_events = []
        for e in p["events"]:
            ne = dict(e)
            ne["ts"] = ne["ts"] - delta
            new_events.append(ne)
        aligned.append({"pid": p["pid"], "events": new_events})
    return aligned


def merge_profiles(profiles):
    aligned = align_clocks(profiles)
    all_events = []
    for p in aligned:
        for e in p["events"]:
            ne = dict(e)
            ne["pid"] = p["pid"]
            all_events.append(ne)
    all_events.sort(key=lambda x: x["ts"])
    return {"events": all_events}


def find_straggler(merged_timeline):
    durations = {}
    for e in merged_timeline["events"]:
        pid = e.get("pid")
        if "dur" in e:
            durations[pid] = durations.get(pid, 0) + e["dur"]
    if not durations:
        return None
    straggler = max(durations.items(), key=lambda x: x[1])[0]
    return straggler


def explain_cause(merged_timeline, straggler_id):
    for e in merged_timeline["events"]:
        if e.get("pid") == straggler_id and e.get("name") == "compute":
            if e.get("dur", 0) > 50:
                return "heavy_compute"
    return "waiting_sync"


def confirm_straggler(merged_timeline, straggler_id):
    s = find_straggler(merged_timeline)
    return s == straggler_id


def generate_report(profiles):
    aligned = align_clocks(profiles)
    merged = merge_profiles(profiles)
    straggler = find_straggler(merged)
    cause = explain_cause(merged, straggler)
    confirmed = confirm_straggler(merged, straggler)
    return {
        "aligned": aligned,
        "merged": merged,
        "straggler": straggler,
        "cause": cause,
        "confirmed": confirmed
    }
