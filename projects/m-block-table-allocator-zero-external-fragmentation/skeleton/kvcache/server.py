from .allocator import BlockAllocator


class PrefixCache:
    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError()

    def get_or_allocate(self, tokens: list[int]) -> tuple[int, int]:
        raise NotImplementedError()


def measure_hit_rate(traces: list[list[int]], num_blocks: int, block_size: int) -> float:
    raise NotImplementedError()
