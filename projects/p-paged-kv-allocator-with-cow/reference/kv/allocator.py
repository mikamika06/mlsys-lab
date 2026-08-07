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
        return len(self.free_blocks)

    def allocate_sequence(self) -> int:
        sid = self.next_seq_id
        self.next_seq_id += 1
        self.seq_lengths[sid] = 0
        self.block_tables[sid] = []
        return sid

    def get_block_table(self, seq_id: int) -> list[int]:
        return self.block_tables[seq_id].copy()

    def get_block_refcount(self, block_id: int) -> int:
        return self.ref_counts[block_id]

    def append_tokens(self, seq_id: int, num_tokens: int):
        for _ in range(num_tokens):
            curr_len = self.seq_lengths[seq_id]
            block_idx = curr_len // self.block_size
            if block_idx == len(self.block_tables[seq_id]):
                if not self.free_blocks:
                    raise RuntimeError("OOM")
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                self.block_tables[seq_id].append(b)
            else:
                b = self.block_tables[seq_id][block_idx]
                if self.ref_counts[b] > 1:
                    if not self.free_blocks:
                        raise RuntimeError("OOM")
                    b_new = self.free_blocks.pop()
                    self.ref_counts[b] -= 1
                    self.ref_counts[b_new] = 1
                    self.block_tables[seq_id][block_idx] = b_new
            self.seq_lengths[seq_id] += 1

    def fork_sequence(self, parent_id: int) -> int:
        sid = self.next_seq_id
        self.next_seq_id += 1
        self.seq_lengths[sid] = self.seq_lengths[parent_id]
        self.block_tables[sid] = self.block_tables[parent_id].copy()
        for b in self.block_tables[sid]:
            self.ref_counts[b] += 1
        return sid

    def free_sequence(self, seq_id: int):
        for b in self.block_tables[seq_id]:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.free_blocks.append(b)
        del self.seq_lengths[seq_id]
        del self.block_tables[seq_id]
