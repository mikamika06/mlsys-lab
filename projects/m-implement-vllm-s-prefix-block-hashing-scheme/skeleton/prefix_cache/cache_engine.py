class PrefixCacheEngine:
    """Manages cached KV blocks mapped by block hash."""

    def __init__(self, block_size, total_blocks):
        raise NotImplementedError

    def allocate_or_get(self, token_ids):
        raise NotImplementedError

    def simulate_request(self, token_ids, computation_time_per_token=0.001):
        raise NotImplementedError
