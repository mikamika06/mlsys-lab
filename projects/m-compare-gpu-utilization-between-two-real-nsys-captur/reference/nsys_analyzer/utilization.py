def compute_gpu_utilization(kernel_events, capture_window):
    start_win, end_win = capture_window
    if end_win <= start_win:
        return 0.0

    intervals = []
    for k in kernel_events:
        ks = max(k["start_ns"], start_win)
        ke = min(k["end_ns"], end_win)
        if ks < ke:
            intervals.append((ks, ke))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end = intervals[0]

    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))

    active_ns = sum(e - s for s, e in merged)
    total_ns = end_win - start_win
    return (active_ns / total_ns) * 100.0


def compare_batch_utilizations(captures):
    utils = [
        compute_gpu_utilization(cap["kernel_events"], cap["capture_window"])
        for cap in captures
    ]
    min_idx = min(range(len(utils)), key=lambda i: utils[i]) if utils else 0
    return {"utilizations": utils, "argmin_index": min_idx}
