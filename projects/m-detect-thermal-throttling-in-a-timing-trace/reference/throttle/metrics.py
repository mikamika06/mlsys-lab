"""Metrics computation for timing traces."""
import numpy as np


def compute_trace_metrics(durations, events, window_size=5):
    arr = np.asarray(durations, dtype=np.float64)
    n = len(arr)

    throttled_mask = np.zeros(n, dtype=bool)
    for start, end in events:
        throttled_mask[start:end + 1] = True

    unthrottled_indices = np.where(~throttled_mask)[0]
    if len(unthrottled_indices) == 0:
        baseline_median = float(np.median(arr))
    else:
        baseline_median = float(np.median(arr[unthrottled_indices]))

    throttled_indices = np.where(throttled_mask)[0]
    if len(throttled_indices) == 0:
        peak_slowdown = 1.0
        mean_slowdown = 1.0
    else:
        peak_slowdown = float(np.max(arr[throttled_indices]) / baseline_median)
        mean_slowdown = float(np.mean(arr[throttled_indices]) / baseline_median)

    total_time = float(np.sum(arr))
    throttled_time = float(np.sum(arr[throttled_indices])) if len(throttled_indices) > 0 else 0.0
    throttled_fraction = throttled_time / total_time if total_time > 0 else 0.0

    return {
        "baseline_median": baseline_median,
        "peak_slowdown": peak_slowdown,
        "mean_slowdown": mean_slowdown,
        "throttled_fraction": throttled_fraction,
        "num_events": len(events)
    }
