class CacheMetricsTracker:
    """Tracks hit rate and TTFT metrics across requests."""

    def __init__(self):
        raise NotImplementedError

    def record_request(self, total_tokens, cached_tokens, ttft_ms):
        raise NotImplementedError

    def get_hit_rate(self):
        raise NotImplementedError

    def get_summary(self):
        raise NotImplementedError
