class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        raise NotImplementedError

    def alloc(self) -> int:
        raise NotImplementedError

    def free(self, block_id: int) -> None:
        raise NotImplementedError

    def ref(self, block_id: int) -> None:
        raise NotImplementedError

    def deref(self, block_id: int) -> int:
        raise NotImplementedError

    def get_ref_count(self, block_id: int) -> int:
        raise NotImplementedError

class SequenceManager:
    def __init__(self, allocator: BlockAllocator, block_size: int):
        raise NotImplementedError

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        raise NotImplementedError

    def append_token(self, seq_id: int) -> int:
        raise NotImplementedError

    def free_sequence(self, seq_id: int) -> None:
        raise NotImplementedError

    def get_block_table(self, seq_id: int) -> list[int]:
        raise NotImplementedError

def compute_slot_mapping(block_tables: list[list[int]], seq_lens: list[int], block_size: int) -> list[int]:
    raise NotImplementedError
