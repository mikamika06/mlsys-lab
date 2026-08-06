from nsysprof.parser import compute_gpu_idle_gap_pct


def compute_warmup_metrics(kernels):
    if not kernels:
        return 100.0, 0.0
    wall_start = kernels[0]["start"]
    wall_end = kernels[-1]["end"]
    idle_pct = compute_gpu_idle_gap_pct(kernels, wall_start, wall_end)
    util_pct = 100.0 - idle_pct
    return idle_pct, util_pct
