class KVAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.ref_counts = {i: 0 for i in range(num_blocks)}
        self.seq_lengths = {}
        self.block_tables = {}
        self.next_seq_id = 1

    def free_count(self) -> int:
        raise NotImplementedError

    def allocate_sequence(self) -> int:
        raise NotImplementedError

    def get_block_table(self, seq_id: int) -> list[int]:
        raise NotImplementedError

    def get_block_refcount(self, block_id: int) -> int:
        raise NotImplementedError

    def append_tokens(self, seq_id: int, num_tokens: int):
        raise NotImplementedError

    def fork_sequence(self, parent_id: int) -> int:
        raise NotImplementedError

    def free_sequence(self, seq_id: int):
        raise NotImplementedError
