class BlockAllocator:
    def __init__(self, num_blocks: int):
        raise NotImplementedError()

    def allocate(self) -> int:
        raise NotImplementedError()

    def free(self, block_id: int):
        raise NotImplementedError()
