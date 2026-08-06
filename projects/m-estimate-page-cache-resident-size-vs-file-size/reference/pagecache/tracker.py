class PageCacheTracker:
    """Track page residency over access and eviction events."""

    def __init__(self, file_size, page_size=4096):
        self.file_size = file_size
        self.page_size = page_size
        self.max_pages = (file_size + page_size - 1) // page_size
        self.resident_pages = set()

    def access(self, accesses):
        for offset, length in accesses:
            if length <= 0 or offset >= self.file_size:
                continue
            end_offset = min(offset + length, self.file_size)
            start_p = offset // self.page_size
            end_p = (end_offset - 1) // self.page_size
            for p in range(start_p, end_p + 1):
                if p < self.max_pages:
                    self.resident_pages.add(p)

    def evict(self, page_indices):
        for p in page_indices:
            self.resident_pages.discard(p)

    def get_resident_bytes(self):
        return len(self.resident_pages) * self.page_size
