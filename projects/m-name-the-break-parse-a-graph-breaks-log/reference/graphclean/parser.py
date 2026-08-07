def parse_graph_breaks(log_text):
    lines = [l.strip() for l in log_text.strip().splitlines() if l.strip()]
    counts = {}
    for line in lines:
        if "reason:" in line:
            reason = line.split("reason:")[1].strip().rstrip(")")
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "total_breaks": len(lines),
        "unique_types": len(counts),
        "break_counts": counts
    }
