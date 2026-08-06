class RSSTracker:
    """Track and compare RSS growth over N tokens with vs without set_cache_limit."""

    def __init__(self, hw_memsize_bytes: int):
        raise NotImplementedError

    def simulate_generation(self, num_tokens: int, token_alloc_bytes: int, use_cache_limit: bool) -> list:
        raise NotImplementedError

    def compare_rss_growth(self, num_tokens: int, token_alloc_bytes: int) -> dict:
        raise NotImplementedError
