import math


def compute_physical_blocks_needed(seq_lens: list[int], block_size: int) -> dict[str, float]:
    """
    Computes total physical blocks, total token capacity, total used tokens,
    and internal fragmentation ratio across a batch of sequence lengths.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    total_blocks = 0
    total_used_tokens = 0

    for l in seq_lens:
        if l > 0:
            blocks = math.ceil(l / block_size)
            total_blocks += blocks
            total_used_tokens += l

    total_capacity = total_blocks * block_size
    unused_tokens = total_capacity - total_used_tokens
    frag_ratio = unused_tokens / total_capacity if total_capacity > 0 else 0.0

    return {
        "total_blocks": float(total_blocks),
        "total_capacity": float(total_capacity),
        "total_used_tokens": float(total_used_tokens),
        "fragmentation_ratio": float(frag_ratio),
    }


class BlockTableAllocator:
    """
    Minimal PagedAttention physical block allocator managing a free pool.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_pool = list(range(num_blocks))
        self.tables: dict[int, list[int]] = {}
        self.seq_lens: dict[int, int] = {}

    def allocate(self, seq_id: int, initial_seq_len: int) -> list[int]:
        if seq_id in self.tables:
            raise ValueError(f"Sequence {seq_id} already allocated")
        needed_blocks = math.ceil(initial_seq_len / self.block_size) if initial_seq_len > 0 else 0
        if len(self.free_pool) < needed_blocks:
            raise MemoryError("Out of physical KV blocks")

        allocated = [self.free_pool.pop(0) for _ in range(needed_blocks)]
        self.tables[seq_id] = allocated
        self.seq_lens[seq_id] = initial_seq_len
        return list(allocated)

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> list[int]:
        if seq_id not in self.tables:
            raise KeyError(f"Sequence {seq_id} not found")

        current_len = self.seq_lens[seq_id]
        new_len = current_len + num_new_tokens
        current_blocks = len(self.tables[seq_id])
        needed_blocks = math.ceil(new_len / self.block_size) if new_len > 0 else 0

        additional = needed_blocks - current_blocks
        if additional > 0:
            if len(self.free_pool) < additional:
                raise MemoryError("Out of physical KV blocks during token append")
            for _ in range(additional):
                self.tables[seq_id].append(self.free_pool.pop(0))

        self.seq_lens[seq_id] = new_len
        return list(self.tables[seq_id])

    def free(self, seq_id: int) -> None:
        if seq_id in self.tables:
            blocks = self.tables.pop(seq_id)
            del self.seq_lens[seq_id]
            self.free_pool.extend(blocks)

    def get_block_table(self, seq_id: int) -> list[int]:
        if seq_id not in self.tables:
            raise KeyError(f"Sequence {seq_id} not found")
        return list(self.tables[seq_id])

    @property
    def free_blocks_count(self) -> int:
        return len(self.free_pool)
