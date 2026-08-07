class PagedKVAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list = list(range(num_blocks - 1, -1, -1))
        self.ref_count = {i: 0 for i in range(num_blocks)}
        self.block_tables = {}
        self.seq_lengths = {}

    def free_count(self) -> int:
        raise NotImplementedError

    def allocate_seq(self, seq_id: int):
        raise NotImplementedError

    def append_token(self, seq_id: int) -> int:
        raise NotImplementedError

    def fork_seq(self, parent_id: int, child_id: int):
        raise NotImplementedError

    def free_seq(self, seq_id: int):
        raise NotImplementedError

    def get_block_table(self, seq_id: int) -> list[int]:
        raise NotImplementedError
