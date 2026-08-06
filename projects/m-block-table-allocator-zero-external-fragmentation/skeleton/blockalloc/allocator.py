class BlockTableAllocator:
    """Paged block-table allocator guaranteeing zero external fragmentation."""

    def __init__(self, num_blocks, block_size):
        raise NotImplementedError

    def allocate(self, seq_id, num_tokens):
        raise NotImplementedError

    def free(self, seq_id):
        raise NotImplementedError

    def get_block_table(self, seq_id):
        raise NotImplementedError

    def external_fragmentation(self):
        raise NotImplementedError
