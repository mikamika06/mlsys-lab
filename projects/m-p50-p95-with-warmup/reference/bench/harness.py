import numpy as np
from bench.coldstart import compute_cold_start_inflation


def benchmark_with_outlier_rejection(latencies, warmup_count, mad_threshold=3.0):
    """Filters steady-state outliers using Median Absolute Deviation and calculates metrics."""
    arr = np.asarray(latencies, dtype=np.float64)
    cold_info = compute_cold_start_inflation(arr, warmup_count)
    if len(arr) <= warmup_count:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "inflation_ratio": 1.0,
            "retained_count": 0,
        }
    steady = arr[warmup_count:].copy()
    med = np.median(steady)
    mad = np.median(np.abs(steady - med))
    if mad > 1e-9:
        mod_z = 0.6745 * np.abs(steady - med) / mad
        clean = steady[mod_z <= mad_threshold]
    else:
        clean = steady
    if len(clean) == 0:
        clean = steady
    p50 = float(np.percentile(clean, 50))
    p95 = float(np.percentile(clean, 95))
    return {
        "p50": p50,
        "p95": p95,
        "inflation_ratio": cold_info["inflation_ratio"],
        "retained_count": int(len(clean)),
    }
