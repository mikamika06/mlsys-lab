def get_optimizer_durations(events):
    return [e["dur"] for e in events if e["name"] == "optimizer_step"]

def get_spike_index(durations):
    if not durations:
        return -1
    m = max(durations)
    for i, d in enumerate(durations):
        if d == m:
            return i
    return -1

def calculate_spillover_bytes(events, step_start, step_end):
    total = 0
    for e in events:
        if e["name"] in ("cudaMemcpyH2D", "cudaMemcpyD2H"):
            if e["ts"] >= step_start and e["ts"] < step_end:
                total += e.get("args", {}).get("bytes", 0)
    return total
