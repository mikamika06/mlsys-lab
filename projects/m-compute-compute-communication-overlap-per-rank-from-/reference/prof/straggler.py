def identify_straggler(trace_data):
    rank_totals = {}
    for rank, events in trace_data.items():
        total = sum(
            ev["end"] - ev["start"]
            for ev in events
            if ev.get("type") == "allreduce"
        )
        rank_totals[rank] = total
    if not rank_totals:
        return None
    return max(rank_totals, key=rank_totals.get)
