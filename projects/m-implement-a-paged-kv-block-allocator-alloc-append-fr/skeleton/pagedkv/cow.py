class Sequence:
    """Represents a sequence with block table."""
    def __init__(self, seq_id: int, block_size: int):
        raise NotImplementedError

class SequenceManager:
    """Manages sequences and copy-on-write."""
    def __init__(self, allocator):
        raise NotImplementedError

    def create_sequence(self, seq_id: int, block_size: int):
        raise NotImplementedError

    def append_tokens(self, seq: Sequence, num_tokens: int):
        raise NotImplementedError

    def fork_sequence(self, parent_id: int, child_id: int) -> Sequence:
        raise NotImplementedError
