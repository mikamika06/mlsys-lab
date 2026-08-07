import numpy as np


def compute_percentiles(latencies_ms):
    """Compute p50, p95, and p99 from a list/array of latencies in milliseconds."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return {"p50": p50, "p95": p95, "p99": p99}


def analyze_batch_latencies(batch_profile_data):
    """Compute stats for each batch size in batch_profile_data dict."""
    res = {}
    for b, lats in batch_profile_data.items():
        res[int(b)] = compute_percentiles(lats)
    return res
