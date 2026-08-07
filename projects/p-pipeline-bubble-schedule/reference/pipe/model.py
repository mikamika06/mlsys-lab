def gpipe_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    p = float(num_stages)
    m = float(num_microbatches)
    return (p - 1.0) / (m + p - 1.0)

def gpipe_peak_activation_units(num_stages: int, num_microbatches: int, stage_idx: int) -> int:
    if stage_idx == 0:
        return num_microbatches
    return num_microbatches - stage_idx

def calculate_schedule_metrics(schedule_events: list, num_stages: int, num_microbatches: int) -> dict:
    if not schedule_events:
        return {"makespan": 0, "utilization": 0.0, "bubble_fraction": 1.0}
    max_time = max(ev["start"] + ev["duration"] for ev in schedule_events)
    total_compute = sum(ev["duration"] for ev in schedule_events)
    ideal_compute = num_stages * max_time
    util = total_compute / ideal_compute if ideal_compute > 0 else 0.0
    return {
        "makespan": max_time,
        "utilization": util,
        "bubble_fraction": 1.0 - util
    }
