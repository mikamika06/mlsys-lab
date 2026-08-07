import numpy as np


class BlockAllocator:

    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def allocate(self) -> int:
        raise NotImplementedError

    def free_block(self, block_id: int) -> None:
        raise NotImplementedError

    def fork(self, block_table: list[int]) -> list[int]:
        raise NotImplementedError

    def free_chain(self, block_table: list[int]) -> None:
        raise NotImplementedError

    def append_token(self, block_table: list[int], token_id: int, seq_len: int) -> list[int]:
        raise NotImplementedError
