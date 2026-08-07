class PagedAllocator:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))[::-1]
        self.ref_counts = [0] * num_blocks
        self.blocks = [[] for _ in range(num_blocks)]
        self.block_tables = {}

    def alloc(self, seq_id, tokens):
        self.block_tables[seq_id] = []
        self.append(seq_id, tokens)

    def append(self, seq_id, tokens):
        table = self.block_tables[seq_id]
        for t in tokens:
            if not table:
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                table.append(b)
            last_b = table[-1]
            if len(self.blocks[last_b]) == self.block_size:
                b = self.free_blocks.pop()
                self.ref_counts[b] = 1
                table.append(b)
                last_b = b
            elif self.ref_counts[last_b] > 1:
                new_b = self.free_blocks.pop()
                self.ref_counts[new_b] = 1
                self.blocks[new_b] = list(self.blocks[last_b])
                self.ref_counts[last_b] -= 1
                table[-1] = new_b
                last_b = new_b
            self.blocks[last_b].append(t)

    def fork(self, parent_id, child_id):
        self.block_tables[child_id] = list(self.block_tables[parent_id])
        for b in self.block_tables[child_id]:
            self.ref_counts[b] += 1

    def free(self, seq_id):
        if seq_id not in self.block_tables:
            return
        for b in self.block_tables[seq_id]:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                self.blocks[b] = []
                self.free_blocks.append(b)
        del self.block_tables[seq_id]

    def reconstruct(self, seq_id):
        res = []
        for b in self.block_tables.get(seq_id, []):
            res.extend(self.blocks[b])
        return res
