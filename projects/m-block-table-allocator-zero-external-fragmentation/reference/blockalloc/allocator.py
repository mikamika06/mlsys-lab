class BlockTableAllocator:
    """Paged block-table allocator guaranteeing zero external fragmentation."""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}
        self.seq_lengths = {}

    def allocate(self, seq_id, num_tokens):
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            raise MemoryError("Not enough free blocks available.")
        
        if seq_id in self.block_tables:
            self.free(seq_id)

        allocated = []
        for _ in range(needed_blocks):
            allocated.append(self.free_blocks.pop(0))
        
        self.block_tables[seq_id] = allocated
        self.seq_lengths[seq_id] = num_tokens
        return allocated

    def free(self, seq_id):
        if seq_id in self.block_tables:
            blocks = self.block_tables.pop(seq_id)
            self.free_blocks.extend(blocks)
            self.seq_lengths.pop(seq_id, None)

    def get_block_table(self, seq_id):
        return list(self.block_tables.get(seq_id, []))

    def external_fragmentation(self):
        return 0.0
