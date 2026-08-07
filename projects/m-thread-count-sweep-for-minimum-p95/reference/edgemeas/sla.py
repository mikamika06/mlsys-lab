"""Throughput SLA module."""

from edgemeas.sweep import compute_p95


def max_throughput_under_sla(latency_samples_by_threads, max_allowed_p95):
    best_tp = 0.0
    best_threads = None
    for t, samples in sorted(latency_samples_by_threads.items()):
        p95 = compute_p95(samples)
        if p95 <= max_allowed_p95:
            mean_lat = float(sum(samples)) / len(samples)
            if mean_lat > 0:
                tp = (float(t) / mean_lat) * 1000.0
                if tp > best_tp:
                    best_tp = tp
                    best_threads = t
    return {
        "optimal_threads": best_threads,
        "max_throughput_qps": best_tp,
    }
