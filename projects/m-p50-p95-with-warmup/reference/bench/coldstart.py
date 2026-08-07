import numpy as np
from bench.percentiles import compute_steady_state_percentiles


def compute_cold_start_inflation(latencies, warmup_count):
    """Quantifies cold-start inflation as cold p50 divided by steady-state p50."""
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) <= warmup_count or warmup_count <= 0:
        return {"cold_p50": 0.0, "steady_p50": 0.0, "inflation_ratio": 1.0}
    cold = arr[:warmup_count]
    cold_p50 = float(np.percentile(cold, 50))
    steady_res = compute_steady_state_percentiles(arr, warmup_count)
    steady_p50 = steady_res["p50"]
    ratio = cold_p50 / steady_p50 if steady_p50 > 0 else 1.0
    return {"cold_p50": cold_p50, "steady_p50": steady_p50, "inflation_ratio": float(ratio)}
