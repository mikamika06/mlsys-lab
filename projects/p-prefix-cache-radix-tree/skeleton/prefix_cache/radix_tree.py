class RadixNode:
    """Node in the prefix Radix Tree."""

    def __init__(self, prefix: tuple[int, ...] = ()):
        self.prefix: tuple[int, ...] = prefix
        self.children: dict[int, "RadixNode"] = {}
        self.block_ids: list[int] = []
        self.ref_count: int = 0
        self.last_accessed: float = 0.0


class RadixTree:
    """Radix tree supporting multi-token edge matching and node splitting."""

    def __init__(self):
        self.root = RadixNode()

    def insert(self, chain_hashes: list[int], block_ids: list[int], access_time: float = 0.0) -> None:
        raise NotImplementedError

    def match_prefix(self, chain_hashes: list[int], access_time: float = 0.0) -> tuple[list[int], int]:
        raise NotImplementedError

    def evict_lru(self, num_blocks: int) -> int:
        raise NotImplementedError
