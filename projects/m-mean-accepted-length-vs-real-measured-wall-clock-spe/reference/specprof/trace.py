def parse_trace_events(events):
    """Extracts phase time splits from recorded trace events."""
    cat_durations = {}
    total_dur = 0.0
    for ev in events:
        cat = ev.get("cat", "other")
        dur = float(ev.get("dur", 0.0))
        if cat != "step":
            cat_durations[cat] = cat_durations.get(cat, 0.0) + dur
            total_dur += dur

    if total_dur == 0.0:
        return {"draft": 0.0, "target": 0.0, "verify": 0.0, "overhead": 0.0}

    return {
        "draft": float(cat_durations.get("draft", 0.0) / total_dur),
        "target": float(cat_durations.get("target", 0.0) / total_dur),
        "verify": float(cat_durations.get("verify", 0.0) / total_dur),
        "overhead": float(cat_durations.get("overhead", 0.0) / total_dur),
    }
