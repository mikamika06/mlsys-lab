def compute_percentiles(latencies_ms):
    """Compute p50, p95, and p99 from a list/array of latencies in milliseconds."""
    raise NotImplementedError


def analyze_batch_latencies(batch_profile_data):
    """Compute stats for each batch size in batch_profile_data dict."""
    raise NotImplementedError
