import numpy as np
from latmeas.stats import compute_warmup_percentiles


def measure_cold_start_inflation(latencies, warmup_runs):
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) <= warmup_runs or warmup_runs == 0:
        return {"cold_p50": 0.0, "inflation_ratio": 1.0}
    cold = arr[:warmup_runs]
    cold_p50 = float(np.percentile(cold, 50))
    steady_stats = compute_warmup_percentiles(arr, warmup_runs)
    steady_p50 = steady_stats["p50"]
    ratio = cold_p50 / steady_p50 if steady_p50 > 0 else 1.0
    return {"cold_p50": cold_p50, "inflation_ratio": float(ratio)}
