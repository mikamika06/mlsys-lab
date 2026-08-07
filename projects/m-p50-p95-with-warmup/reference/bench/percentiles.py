import numpy as np


def compute_steady_state_percentiles(latencies, warmup_count):
    """Computes p50 and p95 latencies strictly after excluding warmup iterations."""
    arr = np.asarray(latencies, dtype=np.float64)
    if len(arr) <= warmup_count:
        return {"p50": 0.0, "p95": 0.0}
    steady = arr[warmup_count:]
    p50 = float(np.percentile(steady, 50))
    p95 = float(np.percentile(steady, 95))
    return {"p50": p50, "p95": p95}
