"""Prefix-cache hit-rate simulator."""


class PrefixCacheSimulator:
    """Simulator for block-level LRU prefix caching."""

    def __init__(self, block_size, max_blocks):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.cache = {}
        self.usage = {}
        self.clock = 0
        self.next_block_id = 1
        self.total_hits = 0
        self.total_blocks = 0

    def process_request(self, token_ids):
        """Process a request and return (hits, total_blocks_requested)."""
        num_blocks = len(token_ids) // self.block_size
        if num_blocks == 0:
            return 0, 0

        blocks = []
        for i in range(num_blocks):
            blk_tokens = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            blocks.append(blk_tokens)

        hits = 0
        matched_prefix = True

        for i, blk in enumerate(blocks):
            prefix_key = tuple(blocks[: i + 1])
            self.clock += 1
            if matched_prefix and prefix_key in self.cache:
                hits += 1
                self.usage[prefix_key] = self.clock
            else:
                matched_prefix = False
                if len(self.cache) >= self.max_blocks and prefix_key not in self.cache:
                    lru_key = min(self.usage.keys(), key=lambda k: self.usage[k])
                    del self.cache[lru_key]
                    del self.usage[lru_key]
                if prefix_key not in self.cache:
                    self.cache[prefix_key] = self.next_block_id
                    self.next_block_id += 1
                self.usage[prefix_key] = self.clock

        self.total_hits += hits
        self.total_blocks += num_blocks
        return hits, num_blocks

    def hit_rate(self):
        """Return the overall cache hit rate."""
        if self.total_blocks == 0:
            return 0.0
        return self.total_hits / self.total_blocks
