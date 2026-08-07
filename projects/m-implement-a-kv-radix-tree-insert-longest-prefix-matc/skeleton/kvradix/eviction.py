class EvictableRadixCache:
    """Radix Tree KV Cache with capacity tracking and LRU leaf eviction."""

    def __init__(self, max_tokens):
        """Initialize the cache with a maximum token capacity."""
        raise NotImplementedError

    def inc_ref(self, node):
        """Increment reference count for a node and its ancestors."""
        raise NotImplementedError

    def dec_ref(self, node):
        """Decrement reference count for a node and its ancestors."""
        raise NotImplementedError

    def insert_and_cache(self, tokens, request_id=None):
        """Match and insert tokens, evicting LRU unpinned leaves if capacity exceeded."""
        raise NotImplementedError

    def total_tokens(self):
        """Return total number of tokens stored across all nodes in the tree."""
        raise NotImplementedError
