class PagedBlockAllocator:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.block_tables = {}
        self.free_list = []
        self.num_physical_blocks = 0

    def append(self, seq_id, token_count):
        table = self.block_tables.setdefault(seq_id, [])
        needed = (token_count + self.block_size - 1) // self.block_size
        allocated = []
        while len(table) < needed:
            if self.free_list:
                block_id = self.free_list.pop()
            else:
                block_id = self.num_physical_blocks
                self.num_physical_blocks += 1
            table.append(block_id)
            allocated.append(block_id)
        return allocated

    def free(self, seq_id):
        blocks = self.block_tables.pop(seq_id, [])
        for block_id in blocks:
            self.free_list.append(block_id)
