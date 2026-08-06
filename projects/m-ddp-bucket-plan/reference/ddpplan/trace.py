def _merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_overlap_fraction(events):
    total_comm_time = 0.0
    total_overlap_time = 0.0
    ranks = set(e["rank"] for e in events)
    for r in sorted(ranks):
        rank_events = [e for e in events if e["rank"] == r]
        comm_raw = [(e["start"], e["end"]) for e in rank_events if e["type"] == "comm"]
        compute_raw = [(e["start"], e["end"]) for e in rank_events if e["type"] == "compute"]
        comm_merged = _merge_intervals(comm_raw)
        compute_merged = _merge_intervals(compute_raw)
        for c_s, c_e in comm_merged:
            total_comm_time += (c_e - c_s)
            for p_s, p_e in compute_merged:
                inter_s = max(c_s, p_s)
                inter_e = min(c_e, p_e)
                if inter_e > inter_s:
                    total_overlap_time += (inter_e - inter_s)
    if total_comm_time == 0.0:
        return 0.0
    return total_overlap_time / total_comm_time
