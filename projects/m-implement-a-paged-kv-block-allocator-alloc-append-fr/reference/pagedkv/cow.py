class Sequence:
    """Represents a sequence with block table."""
    def __init__(self, seq_id: int, block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.block_table = []
        self.num_tokens = 0

class SequenceManager:
    """Manages sequences and copy-on-write."""
    def __init__(self, allocator):
        self.allocator = allocator
        self.sequences = {}

    def create_sequence(self, seq_id: int) -> Sequence:
        seq = Sequence(seq_id, self.allocator.block_size)
        self.sequences[seq_id] = seq
        return seq

    def append_tokens(self, seq: Sequence, num_tokens: int):
        for _ in range(num_tokens):
            if not seq.block_table or seq.num_tokens % seq.block_size == 0:
                b = self.allocator.alloc()
                seq.block_table.append(b)
            else:
                last_block = seq.block_table[-1]
                if self.allocator.ref_counts[last_block] > 1:
                    self.allocator.free(last_block)
                    b = self.allocator.alloc()
                    seq.block_table[-1] = b
            seq.num_tokens += 1

    def fork_sequence(self, parent_id: int, child_id: int) -> Sequence:
        parent = self.sequences[parent_id]
        child = Sequence(child_id, parent.block_size)
        child.block_table = list(parent.block_table)
        child.num_tokens = parent.num_tokens
        for b in child.block_table:
            self.allocator.ref(b)
        self.sequences[child_id] = child
        return child
