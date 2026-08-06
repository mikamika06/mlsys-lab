def _merge_intervals(intervals):
    if not intervals:
        return []
    sorted_invs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_invs[0]]
    for current in sorted_invs[1:]:
        prev_start, prev_end = merged[-1]
        if current[0] <= prev_end:
            merged[-1] = (prev_start, max(prev_end, current[1]))
        else:
            merged.append(current)
    return merged

def compute_overlap(trace_data):
    results = {}
    for rank, events in trace_data.items():
        comp_raw = [(ev["start"], ev["end"]) for ev in events if ev.get("type") == "compute"]
        comm_raw = [(ev["start"], ev["end"]) for ev in events if ev.get("type") == "comm"]

        comp_merged = _merge_intervals(comp_raw)
        comm_merged = _merge_intervals(comm_raw)

        comp_duration = sum(e - s for s, e in comp_merged)
        if comp_duration <= 0.0:
            results[rank] = 0.0
            continue

        overlap_duration = 0.0
        for cs, ce in comp_merged:
            for ms, me in comm_merged:
                overlap_duration += max(0.0, min(ce, me) - max(cs, ms))

        pct = (overlap_duration / comp_duration) * 100.0
        results[rank] = float(pct)
    return results
