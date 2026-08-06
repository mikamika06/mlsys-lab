class Block:
    def __init__(self, addr, size, is_allocated, segment_id):
        raise NotImplementedError


class CachingAllocator:
    def __init__(self, segment_size=2097152):
        raise NotImplementedError

    def malloc(self, size: int) -> int:
        raise NotImplementedError

    def free(self, handle: int) -> None:
        raise NotImplementedError

    def coalesce(self) -> None:
        raise NotImplementedError
