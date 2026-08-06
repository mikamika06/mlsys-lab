class PageCacheTracker:
    """Track page residency over access and eviction events."""

    def __init__(self, file_size, page_size=4096):
        raise NotImplementedError

    def access(self, accesses):
        raise NotImplementedError

    def evict(self, page_indices):
        raise NotImplementedError

    def get_resident_bytes(self):
        raise NotImplementedError
