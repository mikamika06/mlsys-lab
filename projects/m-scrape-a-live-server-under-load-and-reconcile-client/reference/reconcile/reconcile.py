import numpy as np


def compute_percentile_error(client_latencies, server_histogram):
    if not client_latencies:
        return 0.0
    arr = np.array(client_latencies)
    client_p99 = np.percentile(arr, 99.0)
    buckets = server_histogram.get("buckets", {})
    total_count = server_histogram.get("count", len(arr))
    if total_count == 0:
        return 1.0
    sorted_le = sorted([k for k in buckets.keys() if k != float("inf")])
    server_p99_est = sorted_le[-1] if sorted_le else client_p99
    for le in sorted_le:
        if buckets[le] / total_count >= 0.99:
            server_p99_est = le
            break
    rel_err = abs(client_p99 - server_p99_est) / max(1e-6, client_p99)
    return float(rel_err)
