class MPSMemoryTracker:
    def __init__(self, recommended_max_bytes: int):
        self.recommended_max_bytes = recommended_max_bytes
        self.live_allocations = {}
        self.cached_freed_blocks = {}

    def allocate(self, block_id: str, size_bytes: int) -> None:
        if block_id in self.live_allocations:
            return
        if block_id in self.cached_freed_blocks:
            cached_size = self.cached_freed_blocks.pop(block_id)
            if cached_size == size_bytes:
                self.live_allocations[block_id] = size_bytes
                return
            else:
                pass
        self.live_allocations[block_id] = size_bytes

    def free(self, block_id: str) -> None:
        if block_id in self.live_allocations:
            sz = self.live_allocations.pop(block_id)
            self.cached_freed_blocks[block_id] = sz

    def empty_cache(self) -> int:
        freed_bytes = sum(self.cached_freed_blocks.values())
        self.cached_freed_blocks.clear()
        return freed_bytes

    def current_allocated_memory(self) -> int:
        return sum(self.live_allocations.values())

    def driver_allocated_memory(self) -> int:
        return sum(self.live_allocations.values()) + sum(self.cached_freed_blocks.values())

    def divergence_bytes(self) -> int:
        return self.driver_allocated_memory() - self.current_allocated_memory()

    def fragmentation_ratio(self) -> float:
        driver = self.driver_allocated_memory()
        if driver == 0:
            return 0.0
        return self.divergence_bytes() / float(driver)
