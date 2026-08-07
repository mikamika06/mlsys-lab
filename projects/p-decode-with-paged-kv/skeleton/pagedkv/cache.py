class BlockCache:
    def __init__(self, block_size, num_blocks, head_dim):
        raise NotImplementedError()
    def allocate(self):
        raise NotImplementedError()
    def release(self, idx):
        raise NotImplementedError()
    def free_count(self):
        raise NotImplementedError()
