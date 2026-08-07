class BlockTable:
    def __init__(self, allocator, blocks=None, data=None):
        self.allocator = allocator
        self.blocks = list(blocks) if blocks is not None else []
        self.data = dict(data) if data is not None else {}

    def append(self, token_val: int):
        logical_idx = len(self.data)
        block_idx = logical_idx // self.allocator.block_size
        if block_idx >= len(self.blocks):
            b = self.allocator.allocate()
            self.blocks.append(b)
        self.data[logical_idx] = token_val

    def write(self, logical_idx: int, token_val: int):
        block_offset = logical_idx // self.allocator.block_size
        phys_block = self.blocks[block_offset]
        if self.allocator.ref_count(phys_block) > 1:
            self.allocator.decref(phys_block)
            new_block = self.allocator.allocate()
            self.blocks[block_offset] = new_block
        self.data[logical_idx] = token_val

    def fork(self) -> "BlockTable":
        for b in self.blocks:
            self.allocator.incref(b)
        return BlockTable(self.allocator, list(self.blocks), dict(self.data))

    def get_blocks(self) -> list[int]:
        return list(self.blocks)
