import numpy as np


def compute_warmup_percentiles(latencies, warmup_runs):
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) <= warmup_runs:
        return {"p50": 0.0, "p95": 0.0}
    steady = arr[warmup_runs:]
    p50 = float(np.percentile(steady, 50))
    p95 = float(np.percentile(steady, 95))
    return {"p50": p50, "p95": p95}
