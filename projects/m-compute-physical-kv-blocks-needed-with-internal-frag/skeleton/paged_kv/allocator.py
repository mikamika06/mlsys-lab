import math


def compute_physical_blocks_needed(seq_lens: list[int], block_size: int) -> dict[str, float]:
    """
    Computes total physical blocks, total token capacity, total used tokens,
    and internal fragmentation ratio across a batch of sequence lengths.
    """
    raise NotImplementedError


class BlockTableAllocator:
    """
    Minimal PagedAttention physical block allocator managing a free pool.
    """

    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def allocate(self, seq_id: int, initial_seq_len: int) -> list[int]:
        raise NotImplementedError

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> list[int]:
        raise NotImplementedError

    def free(self, seq_id: int) -> None:
        raise NotImplementedError

    def get_block_table(self, seq_id: int) -> list[int]:
        raise NotImplementedError

    @property
    def free_blocks_count(self) -> int:
        raise NotImplementedError
