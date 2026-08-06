def calculate_summary(traces):
    """Calculates throughput, TTFT percentiles (P50, P90, P99), and overall latency metrics."""
    raise NotImplementedError


def compute_latency_degradation_ratio(low_concurrency_summary, high_concurrency_summary):
    """Computes high_concurrency_p90_ttft / low_concurrency_p90_ttft ratio."""
    raise NotImplementedError
