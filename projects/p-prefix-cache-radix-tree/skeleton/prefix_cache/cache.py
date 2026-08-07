class TreeNode:
    def __init__(self, block_hash: int, block_id: int, parent=None):
        self.block_hash = block_hash
        self.block_id = block_id
        self.parent = parent
        self.children = {}
        self.ref_count = 0
        self.last_access = 0


class PrefixCache:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = TreeNode(0, -1)
        self.nodes = {}
        self.clock = 0
        self.saved_tokens = 0

    def compute_hashes(self, tenant_id: str, blocks: list[tuple[int, ...]]) -> list[int]:
        raise NotImplementedError

    def match(self, tenant_id: str, blocks: list[tuple[int, ...]]) -> list[int]:
        raise NotImplementedError

    def insert(self, tenant_id: str, blocks: list[tuple[int, ...]], block_ids: list[int]):
        raise NotImplementedError

    def inc_ref(self, block_ids: list[int]):
        raise NotImplementedError

    def dec_ref(self, block_ids: list[int]):
        raise NotImplementedError

    def evict(self) -> int | None:
        raise NotImplementedError
