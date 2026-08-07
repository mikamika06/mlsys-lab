class BlockTable:
    def __init__(self, allocator):
        raise NotImplementedError

    def append(self, token_val: int):
        raise NotImplementedError

    def write(self, logical_idx: int, token_val: int):
        raise NotImplementedError

    def fork(self) -> "BlockTable":
        raise NotImplementedError

    def get_blocks(self) -> list[int]:
        raise NotImplementedError
