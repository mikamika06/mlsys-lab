def parse_prometheus_metrics(raw_text: str) -> dict:
    """Parse Prometheus text format into a metric key-value dictionary."""
    raise NotImplementedError


def analyze_prompt_cache(raw_text: str) -> dict:
    """Analyze prompt cache metrics to evaluate total, processed, and cached tokens."""
    raise NotImplementedError
