def count_gaps(events: list, threshold_ms: float) -> int:
    gaps = 0
    sorted_ev = sorted(events, key=lambda x: x["timestamp"])
    for i in range(len(sorted_ev) - 1):
        dur = (sorted_ev[i+1]["timestamp"] - sorted_ev[i]["timestamp"]) * 1000.0
        if dur > threshold_ms:
            gaps += 1
    return gaps
