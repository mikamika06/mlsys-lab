class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def alloc(self) -> int:
        if not self.free_list:
            raise RuntimeError("Out of blocks")
        b = self.free_list.pop(0)
        self.ref_counts[b] = 1
        return b

    def free(self, block_id: int) -> None:
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            self.ref_counts[block_id] = 0
            self.free_list.append(block_id)

    def ref(self, block_id: int) -> None:
        self.ref_counts[block_id] += 1

    def deref(self, block_id: int) -> int:
        self.ref_counts[block_id] -= 1
        return self.ref_counts[block_id]

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts[block_id]

class SequenceManager:
    def __init__(self, allocator: BlockAllocator, block_size: int):
        self.allocator = allocator
        self.block_size = block_size
        self.seq_block_tables = {}
        self.seq_lengths = {}

    def create_sequence(self, seq_id: int, initial_len: int = 0):
        self.seq_block_tables[seq_id] = []
        self.seq_lengths[seq_id] = initial_len
        num_blocks = (initial_len + self.block_size - 1) // self.block_size if initial_len > 0 else 0
        for _ in range(num_blocks):
            b = self.allocator.alloc()
            self.seq_block_tables[seq_id].append(b)

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        parent_table = self.seq_block_tables[parent_seq_id]
        for b in parent_table:
            self.allocator.ref(b)
        self.seq_block_tables[child_seq_id] = list(parent_table)
        self.seq_lengths[child_seq_id] = self.seq_lengths[parent_seq_id]

    def append_token(self, seq_id: int) -> int:
        table = self.seq_block_tables[seq_id]
        length = self.seq_lengths[seq_id]
        if length == 0 or length % self.block_size == 0:
            b = self.allocator.alloc()
            table.append(b)
        else:
            last_block = table[-1]
            if self.allocator.ref_counts[last_block] > 1:
                self.allocator.free(last_block)
                new_b = self.allocator.alloc()
                table[-1] = new_b
        self.seq_lengths[seq_id] = length + 1
        return self.seq_lengths[seq_id]

    def free_sequence(self, seq_id: int) -> None:
        table = self.seq_block_tables.get(seq_id, [])
        for b in table:
            self.allocator.free(b)
        if seq_id in self.seq_block_tables:
            del self.seq_block_tables[seq_id]
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.seq_block_tables.get(seq_id, [])

def compute_slot_mapping(block_tables: list[list[int]], seq_lens: list[int], block_size: int) -> list[int]:
    slot_mapping = []
    for table, length in zip(block_tables, seq_lens):
        for i in range(length):
            block_idx = i // block_size
            block_offset = i % block_size
            physical_block = table[block_idx]
            slot = physical_block * block_size + block_offset
            slot_mapping.append(slot)
    return slot_mapping
