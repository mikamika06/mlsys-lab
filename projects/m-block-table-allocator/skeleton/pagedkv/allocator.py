class BlockAllocator:
    """Manages physical block allocation and per-sequence block tables."""

    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def allocate((self, seq_id: str, num_tokens: int) -> list[int]:
        raise NotImplementedError

    def append_slots(self, seq_id: str, num_tokens: int) -> list[int]:
        raise NotImplementedError

    def free(self, seq_id: str) -> None:
        raise NotImplementedError

    def get_block_table(self, seq_id: str) -> list[int]:
        raise NotImplementedError

    def get_num_free_blocks(self) -> int:
        raise NotImplementedError
