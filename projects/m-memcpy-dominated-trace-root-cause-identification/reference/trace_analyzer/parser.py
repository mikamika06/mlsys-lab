def parse_events(events: list[dict]) -> dict[str, dict]:
    out = {}
    for ev in events:
        if ev.get("cat") == "Node" and ev.get("ph") == "X":
            op = ev.get("args", {}).get("op_name", "Unknown")
            dur = ev.get("dur", 0.0)
            if op not in out:
                out[op] = {"count": 0, "duration_us": 0.0}
            out[op]["count"] += 1
            out[op]["duration_us"] += dur
    return out
