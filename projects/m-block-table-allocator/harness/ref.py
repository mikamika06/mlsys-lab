import math
import numpy as np


class ReferenceAllocator:
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


def reference_gather_kv_cache(
    physical_blocks: np.ndarray,
    block_table: list[int],
    seq_len: int,
) -> np.ndarray:
    if seq_len == 0 or not block_table:
        shape = (0,) + physical_blocks.shape[2:]
        return np.empty(shape, dtype=physical_blocks.dtype)
    gathered_blocks = physical_blocks[block_table]
    logical_tensor = gathered_blocks.reshape((-1,) + physical_blocks.shape[2:])
    return logical_tensor[:seq_len]


def reference_compute_fragmentation(allocator, seq_lengths: dict[str, int]) -> dict[str, float]:
    total_allocated_slots = 0
    total_used_slots = 0

    for seq_id, seq_len in seq_lengths.items():
        bt = allocator.get_block_table(seq_id)
        allocated_slots = len(bt) * allocator.block_size
        total_allocated_slots += allocated_slots
        total_used_slots += seq_len

    if total_allocated_slots == 0:
        internal_frag = 0.0
    else:
        internal_frag = (total_allocated_slots - total_used_slots) / total_allocated_slots

    free_blocks = allocator.get_num_free_blocks()
    free_slots = free_blocks * allocator.block_size
    total_capacity = allocator.num_blocks * allocator.block_size

    if total_capacity == 0:
        external_frag = 0.0
    else:
        external_frag = free_slots / total_capacity

    return {
        "internal_fragmentation": float(internal_frag),
        "external_fragmentation": float(external_frag),
    }
