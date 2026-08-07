import numpy as np

CONFIGS = [
    {"concurrency": 2, "num_requests": 50, "seed": 42},
    {"concurrency": 4, "num_requests": 100, "seed": 123},
    {"concurrency": 8, "num_requests": 200, "seed": 999},
]

def generate_schedule(concurrency, num_requests, seed=42):
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / concurrency, size=num_requests)
    return np.cumsum(inter_arrivals)

def compute_metrics(latencies, duration):
    latencies = np.asarray(latencies, dtype=float)
    if len(latencies) == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "throughput": 0.0}
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p99 = float(np.percentile(latencies, 99))
    throughput = float(len(latencies) / duration) if duration > 0 else 0.0
    return {"p50": p50, "p90": p90, "p99": p99, "throughput": throughput}
