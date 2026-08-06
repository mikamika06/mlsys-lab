from .allocator import BlockAllocator


class PrefixCache:
    def __init__(self, num_blocks: int, block_size: int):
        self.allocator = BlockAllocator(num_blocks)
        self.block_size = block_size
        self.cache = {}

    def get_or_allocate(self, tokens: list[int]) -> tuple[int, int]:
        hits = 0
        misses = 0
        parent = -1
        num_full = len(tokens) // self.block_size
        for i in range(num_full):
            chunk = tuple(tokens[i * self.block_size : (i + 1) * self.block_size])
            key = (parent, chunk)
            if key in self.cache:
                hits += 1
                parent = self.cache[key]
            else:
                try:
                    phys = self.allocator.allocate()
                except MemoryError:
                    break
                self.cache[key] = phys
                parent = phys
                misses += 1
        return hits, misses


def measure_hit_rate(traces: list[list[int]], num_blocks: int, block_size: int) -> float:
    pc = PrefixCache(num_blocks, block_size)
    total_hits = 0
    total_misses = 0
    for trace in traces:
        h, m = pc.get_or_allocate(trace)
        total_hits += h
        total_misses += m
    if total_hits + total_misses == 0:
        return 0.0
    return total_hits / (total_hits + total_misses)
