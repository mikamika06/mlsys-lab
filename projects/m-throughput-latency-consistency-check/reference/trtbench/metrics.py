import numpy as np


def compute_latency_stats(durations_ms):
    arr = np.asfarray(durations_ms)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr)),
    }


def check_throughput_consistency(durations_ms, total_wall_time_ms, batch_size):
    arr = np.asfarray(durations_ms)
    total_samples = len(arr) * batch_size
    empirical_throughput = (total_samples / total_wall_time_ms) * 1000.0
    sum_latency_s = np.sum(arr) / 1000.0
    implied_throughput = (
        (len(arr) * batch_size) / sum_latency_s if sum_latency_s > 0 else 0.0
    )
    concurrency_factor = (
        empirical_throughput / implied_throughput
        if implied_throughput > 0
        else 0.0
    )
    discrepancy_ratio = abs(empirical_throughput - implied_throughput) / max(
        empirical_throughput, 1e-9
    )
    return {
        "empirical_throughput": float(empirical_throughput),
        "implied_throughput": float(implied_throughput),
        "concurrency_factor": float(concurrency_factor),
        "discrepancy_ratio": float(discrepancy_ratio),
        "is_consistent": bool(discrepancy_ratio < 0.15),
    }
