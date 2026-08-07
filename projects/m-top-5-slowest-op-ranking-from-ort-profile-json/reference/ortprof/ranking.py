def rank_top_slowest(events):
    totals = {}
    for e in events:
        if e.get("cat") == "Node":
            op = e.get("args", {}).get("op_name", "Unknown")
            totals[op] = totals.get(op, 0) + e.get("dur", 0)
    sorted_ops = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [op for op, dur in sorted_ops[:5]]
