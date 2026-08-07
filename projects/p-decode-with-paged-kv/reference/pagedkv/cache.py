import numpy as np

class BlockCache:
    def __init__(self, block_size, num_blocks, head_dim):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.head_dim = head_dim
        self.blocks = np.zeros((num_blocks, 2, block_size, head_dim), dtype=np.float32)
        self.free_list = list(range(num_blocks))

    def allocate(self):
        if not self.free_list:
            raise RuntimeError("Out of blocks")
        return self.free_list.pop(0)

    def release(self, idx):
        if idx not in self.free_list:
            self.free_list.append(idx)
            self.free_list.sort()

    def free_count(self):
        return len(self.free_list)
