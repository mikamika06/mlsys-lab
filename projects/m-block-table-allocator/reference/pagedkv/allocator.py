import math


class BlockAllocator:
    """Manages physical block allocation and per-sequence block tables."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.tables = {}

    def allocate(self, seq_id: str, num_tokens: int) -> list[int]:
        if seq_id in self.tables:
            raise ValueError(f"Sequence {seq_id} already allocated")
        blocks_needed = math.ceil(num_tokens / self.block_size) if num_tokens > 0 else 0
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError("Out of physical blocks")
        allocated = [self.free_blocks.pop(0) for _ in range(blocks_needed)]
        self.tables[seq_id] = allocated
        return allocated

    def append_slots(self, seq_id: str, num_tokens: int) -> list[int]:
        if seq_id not in self.tables:
            raise KeyError(f"Sequence {seq_id} not found")
        current_blocks = self.tables[seq_id]
        current_capacity = len(current_blocks) * self.block_size
        blocks_needed = math.ceil(num_tokens / self.block_size) if num_tokens > 0 else 0
        additional_needed = blocks_needed - len(current_blocks)
        if additional_needed > 0:
            if len(self.free_blocks) < additional_needed:
                raise MemoryError("Out of physical blocks")
            new_blocks = [self.free_blocks.pop(0) for _ in range(additional_needed)]
            current_blocks.extend(new_blocks)
        return current_blocks

    def free(self, seq_id: str) -> None:
        if seq_id not in self.tables:
            return
        blocks = self.tables.pop(seq_id)
        self.free_blocks.extend(blocks)

    def get_block_table(self, seq_id: str) -> list[int]:
        return list(self.tables.get(seq_id, []))

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)
