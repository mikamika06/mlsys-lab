class PagedBlockAllocator:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.block_tables = {}
        self.free_list = []
        self.num_physical_blocks = 0

    def append(self, seq_id, token_count):
        # TODO: This allocator incorrectly creates new blocks after frees
        # and never uses the LIFO free list.
        table = self.block_tables.setdefault(seq_id, [])
        needed = (token_count + self.block_size - 1) // self.block_size
        allocated = []
        while len(table) < needed:
            block_id = self.num_physical_blocks
            self.num_physical_blocks += 1
            table.append(block_id)
            allocated.append(block_id)
        return allocated

    def free(self, seq_id):
        self.block_tables.pop(seq_id, None)
