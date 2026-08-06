import time
import numpy as np


def benchmark_mps_vs_eager(
    graph_fn, eager_fn, inputs: tuple, warmup: int = 5, runs: int = 20
) -> dict:
    """Measures latency and speedup ratio of graph execution vs eager execution."""
    if warmup < 0 or runs <= 0:
        raise ValueError("Invalid warmup or runs count")

    for _ in range(warmup):
        graph_fn(*inputs)
        eager_fn(*inputs)

    graph_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        graph_fn(*inputs)
        t1 = time.perf_counter()
        graph_times.append((t1 - t0) * 1000.0)

    eager_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        eager_fn(*inputs)
        t1 = time.perf_counter()
        eager_times.append((t1 - t0) * 1000.0)

    graph_mean = float(np.mean(graph_times))
    eager_mean = float(np.mean(eager_times))
    speedup = eager_mean / graph_mean if graph_mean > 0 else 1.0

    return {
        "graph_latency_ms": graph_mean,
        "eager_latency_ms": eager_mean,
        "speedup": speedup,
        "graph_p50_ms": float(np.median(graph_times)),
        "graph_p95_ms": float(np.percentile(graph_times, 95)),
        "runs": runs,
        "warmup": warmup,
    }
