import numpy as np


class BlockAllocator:

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.refcounts = [0] * num_blocks
        self.free_blocks = list(range(num_blocks - 1, -1, -1))
        self.physical_tokens = np.full((num_blocks, block_size), -1, dtype=np.int64)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of memory: no free blocks")
        block_id = self.free_blocks.pop()
        self.refcounts[block_id] = 1
        return block_id

    def free_block(self, block_id: int) -> None:
        if self.refcounts[block_id] <= 0:
            raise ValueError(f"Block {block_id} is not currently allocated")
        self.refcounts[block_id] -= 1
        if self.refcounts[block_id] == 0:
            self.free_blocks.append(block_id)

    def fork(self, block_table: list[int]) -> list[int]:
        for block_id in block_table:
            self.refcounts[block_id] += 1
        return list(block_table)

    def free_chain(self, block_table: list[int]) -> None:
        for block_id in block_table:
            self.free_block(block_id)

    def append_token(self, block_table: list[int], token_id: int, seq_len: int) -> list[int]:
        table = list(block_table)
        offset = seq_len % self.block_size
        if seq_len > 0 and offset > 0:
            last_block = table[-1]
            if self.refcounts[last_block] > 1:
                new_block = self.allocate()
                self.physical_tokens[new_block, :offset] = self.physical_tokens[last_block, :offset]
                self.free_block(last_block)
                table[-1] = new_block
                last_block = new_block
            self.physical_tokens[last_block, offset] = token_id
        else:
            new_block = self.allocate()
            table.append(new_block)
            self.physical_tokens[new_block, 0] = token_id
        return table
