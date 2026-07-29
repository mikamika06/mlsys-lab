class Allocator:
    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def allocate(self) -> int:
        raise NotImplementedError

    def share(self, block: int) -> int:
        raise NotImplementedError

    def release(self, block: int) -> None:
        raise NotImplementedError

    def free_count(self) -> int:
        raise NotImplementedError

    def register(self, block: int, key: str) -> None:
        raise NotImplementedError

    def lookup(self, key: str):
        raise NotImplementedError
