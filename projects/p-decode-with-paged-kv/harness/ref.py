import numpy as np

def get_reference_cache(block_size, num_blocks, head_dim):
    class RefCache:
        def __init__(self):
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.head_dim = head_dim
            self.blocks = np.zeros((num_blocks, 2, block_size, head_dim), dtype=np.float32)
            self.free_blocks = list(range(num_blocks))
        def allocate(self):
            if not self.free_blocks:
                raise RuntimeError("Out of blocks")
            return self.free_blocks.pop(0)
        def release(self, idx):
            if idx not in self.free_blocks:
                self.free_blocks.append(idx)
                self.free_blocks.sort()
        def free_count(self):
            return len(self.free_blocks)
    return RefCache()

def compute_block_table(seq_lens, block_size):
    tables = []
    for slen in seq_lens:
        n_blocks = max(1, (slen + block_size - 1) // block_size)
        tables.append(list(range(n_blocks)))
    return tables
