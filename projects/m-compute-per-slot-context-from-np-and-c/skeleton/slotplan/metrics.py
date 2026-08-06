def parse_metrics(metrics_text):
    """Parse llama-server /metrics endpoint output."""
    raise NotImplementedError


def compute_cache_reuse_ratio(metrics_text):
    """Compute prompt cache reuse ratio from parsed metrics."""
    raise NotImplementedError
