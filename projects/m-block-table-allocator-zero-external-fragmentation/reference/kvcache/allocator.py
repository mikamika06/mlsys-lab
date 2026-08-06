class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free_list = list(range(num_blocks - 1, -1, -1))

    def allocate(self) -> int:
        if not self.free_list:
            raise MemoryError()
        return self.free_list.pop()

    def free(self, block_id: int):
        self.free_list.append(block_id)
