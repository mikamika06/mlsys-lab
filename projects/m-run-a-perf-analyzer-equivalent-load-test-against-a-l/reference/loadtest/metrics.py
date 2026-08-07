import numpy as np

def compute_metrics(latencies, duration):
    latencies = np.asarray(latencies, dtype=float)
    if len(latencies) == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "throughput": 0.0}
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p99 = float(np.percentile(latencies, 99))
    throughput = float(len(latencies) / duration) if duration > 0 else 0.0
    return {"p50": p50, "p90": p90, "p99": p99, "throughput": throughput}
