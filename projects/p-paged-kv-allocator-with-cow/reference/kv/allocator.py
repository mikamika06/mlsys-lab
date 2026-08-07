class PagedKVAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_list = list(range(num_blocks - 1, -1, -1))
        self.ref_count = {i: 0 for i in range(num_blocks)}
        self.block_tables = {}
        self.seq_lengths = {}

    def free_count(self) -> int:
        return len(self.free_list)

    def allocate_seq(self, seq_id: int):
        if seq_id in self.block_tables:
            raise ValueError("Sequence already exists")
        self.block_tables[seq_id] = []
        self.seq_lengths[seq_id] = 0

    def append_token(self, seq_id: int) -> int:
        length = self.seq_lengths[seq_id]
        blocks = self.block_tables[seq_id]

        if length % self.block_size == 0:
            if not self.free_list:
                raise RuntimeError("OOM")
            new_block = self.free_list.pop()
            self.ref_count[new_block] = 1
            blocks.append(new_block)
            block_idx = new_block
        else:
            last_block = blocks[-1]
            if self.ref_count[last_block] > 1:
                if not self.free_list:
                    raise RuntimeError("OOM")
                new_block = self.free_list.pop()
                self.ref_count[new_block] = 1
                self.ref_count[last_block] -= 1
                blocks[-1] = new_block
                block_idx = new_block
            else:
                block_idx = last_block

        self.seq_lengths[seq_id] += 1
        return block_idx

    def fork_seq(self, parent_id: int, child_id: int):
        if child_id in self.block_tables:
            raise ValueError("Child already exists")
        self.block_tables[child_id] = list(self.block_tables[parent_id])
        self.seq_lengths[child_id] = self.seq_lengths[parent_id]
        for b in self.block_tables[child_id]:
            self.ref_count[b] += 1

    def free_seq(self, seq_id: int):
        if seq_id not in self.block_tables:
            return
        blocks = self.block_tables.pop(seq_id)
        del self.seq_lengths[seq_id]
        for b in blocks:
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_list.append(b)

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.block_tables.get(seq_id, [])
