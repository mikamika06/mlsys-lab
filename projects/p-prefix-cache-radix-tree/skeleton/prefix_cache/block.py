class BlockHasher:
    """Computes individual and chained hashes for token blocks."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size

    def hash_block(self, tokens: list[int]) -> int:
        raise NotImplementedError

    def compute_chain_hashes(self, tokens: list[int], tenant_id: str = "") -> list[int]:
        raise NotImplementedError
