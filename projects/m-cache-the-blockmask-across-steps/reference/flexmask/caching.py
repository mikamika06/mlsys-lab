class BlockMaskCache:
    """Caches BlockMask across steps."""
    def __init__(self):
        self.cache = {}

    def get_or_create(self, seq_len, block_size, factory_fn):
        key = (seq_len, block_size)
        if key not in self.cache:
            self.cache[key] = factory_fn(seq_len, block_size)
        return self.cache[key]
