class MPSMemoryTracker:
    """Tracks current allocated memory vs driver allocated memory."""

    def __init__(self, recommended_max_bytes: int):
        raise NotImplementedError

    def allocate(self, block_id: str, size_bytes: int) -> None:
        raise NotImplementedError

    def free(self, block_id: str) -> None:
        raise NotImplementedError

    def current_allocated_memory(self) -> int:
        raise NotImplementedError

    def driver_allocated_memory(self) -> int:
        raise NotImplementedError

    def divergence_bytes(self) -> int:
        raise NotImplementedError

    def fragmentation_ratio(self) -> float:
        raise NotImplementedError
