class CacheMetricsTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.requests = []

    def record_request(self, total_tokens, cached_tokens, ttft_ms):
        self.total_prompt_tokens += total_tokens
        self.total_cached_tokens += cached_tokens
        self.requests.append(
            {"total": total_tokens, "cached": cached_tokens, "ttft": ttft_ms}
        )

    def get_hit_rate(self):
        if self.total_prompt_tokens == 0:
            return 0.0
        return float(self.total_cached_tokens) / float(self.total_prompt_tokens)

    def get_summary(self):
        return {
            "hit_rate": self.get_hit_rate(),
            "total_tokens": float(self.total_prompt_tokens),
            "cached_tokens": float(self.total_cached_tokens),
            "num_requests": float(len(self.requests)),
        }
