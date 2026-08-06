class CacheTracker:
    """Tracks block ownership and lookup events across requests."""

    def __init__(self, block_size=16):
        self.block_size = block_size

    def process_event(self, event):
        raise NotImplementedError

    def get_block_owners(self):
        raise NotImplementedError
