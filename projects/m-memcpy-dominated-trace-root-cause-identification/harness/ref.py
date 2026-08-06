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


def compare_profiles(prof_a: dict[str, dict], prof_b: dict[str, dict]) -> list[dict]:
    all_ops = set(prof_a.keys()) | set(prof_b.keys())
    diffs = []
    for op in all_ops:
        ca = prof_a.get(op, {}).get("count", 0)
        da = prof_a.get(op, {}).get("duration_us", 0.0)
        cb = prof_b.get(op, {}).get("count", 0)
        db = prof_b.get(op, {}).get("duration_us", 0.0)
        diffs.append({
            "op_name": op,
            "count_diff": cb - ca,
            "duration_diff": db - da
        })
    return sorted(diffs, key=lambda x: x["duration_diff"])


def generate_trace_o0():
    return [
        {"cat": "Node", "ph": "X", "dur": 1000, "args": {"op_name": "Conv"}},
        {"cat": "Node", "ph": "X", "dur": 2000, "args": {"op_name": "Memcpy"}},
        {"cat": "Node", "ph": "X", "dur": 2000, "args": {"op_name": "Memcpy"}},
        {"cat": "Node", "ph": "X", "dur": 500, "args": {"op_name": "Relu"}},
        {"cat": "Session", "ph": "B", "dur": 100, "args": {}}
    ]


def generate_trace_o99():
    return [
        {"cat": "Node", "ph": "X", "dur": 1200, "args": {"op_name": "FusedConv"}},
        {"cat": "Session", "ph": "E", "dur": 100, "args": {}}
    ]
