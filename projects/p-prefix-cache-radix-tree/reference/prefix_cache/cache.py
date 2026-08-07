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
        hashes = []
        h = 0
        for c in tenant_id:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        for b in blocks:
            for tk in b:
                h = (h * 31 + tk) & 0xFFFFFFFF
            hashes.append(h)
        return hashes

    def match(self, tenant_id: str, blocks: list[tuple[int, ...]]) -> list[int]:
        hashes = self.compute_hashes(tenant_id, blocks)
        self.clock += 1
        curr = self.root
        matched = []
        for h in hashes:
            if h in curr.children:
                curr = curr.children[h]
                curr.last_access = self.clock
                matched.append(curr.block_id)
            else:
                break
        self.saved_tokens += len(matched) * self.block_size
        return matched

    def insert(self, tenant_id: str, blocks: list[tuple[int, ...]], block_ids: list[int]):
        hashes = self.compute_hashes(tenant_id, blocks)
        self.clock += 1
        curr = self.root
        for h, b_id in zip(hashes, block_ids):
            if h in curr.children:
                curr = curr.children[h]
            else:
                new_node = TreeNode(h, b_id, parent=curr)
                new_node.last_access = self.clock
                curr.children[h] = new_node
                self.nodes[b_id] = new_node
                curr = new_node

    def inc_ref(self, block_ids: list[int]):
        for b_id in block_ids:
            if b_id in self.nodes:
                self.nodes[b_id].ref_count += 1

    def dec_ref(self, block_ids: list[int]):
        for b_id in block_ids:
            if b_id in self.nodes:
                self.nodes[b_id].ref_count -= 1

    def evict(self) -> int | None:
        candidates = []
        for b_id, node in self.nodes.items():
            if node.ref_count == 0 and len(node.children) == 0:
                candidates.append(node)
        if not candidates:
            return None

        best = min(candidates, key=lambda n: n.last_access)
        if best.parent:
            best.parent.children.pop(best.block_hash)
        del self.nodes[best.block_id]
        return best.block_id
