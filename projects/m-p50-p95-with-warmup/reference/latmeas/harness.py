import numpy as np
from latmeas.coldstart import measure_cold_start_inflation
from latmeas.stats import compute_warmup_percentiles


def run_benchmark_harness(latencies, warmup_runs, mad_threshold=3.0):
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) <= warmup_runs:
        return {
            "steady_p50": 0.0,
            "steady_p95": 0.0,
            "inflation_ratio": 1.0,
            "clean_count": 0,
        }
    steady = arr[warmup_runs:].copy()
    med = np.median(steady)
    mad = np.median(np.abs(steady - med))
    if mad > 1e-9:
        modified_z = 0.6745 * np.abs(steady - med) / mad
        clean_steady = steady[modified_z <= mad_threshold]
    else:
        clean_steady = steady
    if len(clean_steady) == 0:
        clean_steady = steady
    p50 = float(np.percentile(clean_steady, 50))
    p95 = float(np.percentile(clean_steady, 95))
    cold_info = measure_cold_start_inflation(arr, warmup_runs)
    return {
        "steady_p50": p50,
        "steady_p95": p95,
        "inflation_ratio": cold_info["inflation_ratio"],
        "clean_count": int(len(clean_steady)),
    }
