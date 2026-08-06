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
