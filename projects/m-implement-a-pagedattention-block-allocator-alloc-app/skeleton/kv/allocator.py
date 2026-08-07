class PagedAllocator:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))[::-1]
        self.ref_counts = [0] * num_blocks
        self.blocks = [[] for _ in range(num_blocks)]
        self.block_tables = {}

    def alloc(self, seq_id, tokens):
        raise NotImplementedError

    def append(self, seq_id, tokens):
        raise NotImplementedError

    def fork(self, parent_id, child_id):
        raise NotImplementedError

    def free(self, seq_id):
        raise NotImplementedError

    def reconstruct(self, seq_id):
        raise NotImplementedError
