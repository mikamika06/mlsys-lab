def compute_misattribution(cpu_durations: list, mps_durations: list) -> float:
    if not cpu_durations or not mps_durations:
        return 0.0
    total_cpu = sum(cpu_durations)
    total_mps = sum(mps_durations)
    if total_cpu == 0:
        return 0.0
    return float(total_cpu / (total_cpu + total_mps))
