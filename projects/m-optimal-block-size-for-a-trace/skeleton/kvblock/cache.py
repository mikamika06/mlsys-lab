"""Prefix-cache hit-rate simulator."""


class PrefixCacheSimulator:
    """Simulator for block-level LRU prefix caching."""

    def __init__(self, block_size, max_blocks):
        self.block_size = block_size
        self.max_blocks = max_blocks
        raise NotImplementedError

    def process_request(self, token_ids):
        """Process a request and return (hits, total_blocks_requested)."""
        raise NotImplementedError

    def hit_rate(self):
        """Return the overall cache hit rate."""
        raise NotImplementedError
