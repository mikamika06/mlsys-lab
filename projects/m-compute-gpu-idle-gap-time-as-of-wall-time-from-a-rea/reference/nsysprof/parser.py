def compute_gpu_idle_gap_pct(kernels, wall_start, wall_end):
    if not kernels:
        return 100.0
    sorted_k = sorted(kernels, key=lambda x: x["start"])
    merged = []
    for k in sorted_k:
        start = max(wall_start, k["start"])
        end = min(wall_end, k["end"])
        if start >= end:
            continue
        if not merged:
            merged.append([start, end])
        else:
            if start < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
    active_time = sum(e - s for s, e in merged)
    wall_time = wall_end - wall_start
    if wall_time <= 0:
        return 0.0
    idle_time = wall_time - active_time
    if idle_time < 0:
        idle_time = 0.0
    return (idle_time / wall_time) * 100.0
