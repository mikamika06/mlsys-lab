def reconstruct_nvtx_depths(events):
    """Reconstruct nesting depth timeline from flat push/pop events."""
    sorted_events = sorted(events, key=lambda x: (x["timestamp_ns"], 0 if x["event_type"] == "pop" else 1))
    stack = []
    timeline = []
    current_depth = 0

    for ev in sorted_events:
        ts = ev["timestamp_ns"]
        etype = ev["event_type"]
        name = ev.get("name", "")

        if etype == "push":
            stack.append(name)
            current_depth = len(stack)
            timeline.append({
                "timestamp_ns": ts,
                "event_type": "push",
                "name": name,
                "depth": current_depth,
            })
        elif etype == "pop":
            if not stack:
                raise ValueError("Pop on empty stack")
            popped = stack.pop()
            timeline.append({
                "timestamp_ns": ts,
                "event_type": "pop",
                "name": popped,
                "depth": current_depth,
            })
            current_depth = len(stack)
        else:
            raise ValueError(f"Unknown event type: {etype}")

    if stack:
        raise ValueError(f"Unmatched push events remaining: {len(stack)}")

    return timeline


def analyze_nvtx_nesting(events):
    """Analyze maximum depth and total time spent at each depth."""
    timeline = reconstruct_nvtx_depths(events)
    if not timeline:
        return {"max_depth": 0, "duration_by_depth_ns": {}}

    max_depth = max(ev["depth"] for ev in timeline)
    duration_by_depth = {}

    current_depth = 0
    last_ts = timeline[0]["timestamp_ns"]

    for ev in timeline:
        ts = ev["timestamp_ns"]
        delta = ts - last_ts
        if delta > 0 and current_depth > 0:
            duration_by_depth[current_depth] = duration_by_depth.get(current_depth, 0) + delta

        if ev["event_type"] == "push":
            current_depth = ev["depth"]
        else:
            current_depth = ev["depth"] - 1
        last_ts = ts

    return {
        "max_depth": max_depth,
        "duration_by_depth_ns": duration_by_depth,
    }
