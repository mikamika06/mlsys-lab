from kvradix.radix import RadixTree


class EvictableRadixCache:
    """Radix cache managing token capacity and LRU eviction over unpinned leaves."""

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.tree = RadixTree()
        self.clock = 0

    def inc_ref(self, node):
        curr = node
        while curr is not None and curr is not self.tree.root:
            curr.ref_count += 1
            curr = curr.parent

    def dec_ref(self, node):
        curr = node
        while curr is not None and curr is not self.tree.root:
            curr.ref_count = max(0, curr.ref_count - 1)
            curr = curr.parent

    def _update_access(self, node):
        self.clock += 1
        curr = node
        while curr is not None and curr is not self.tree.root:
            curr.last_accessed = self.clock
            curr = curr.parent

    def total_tokens(self):
        total = 0
        stack = [self.tree.root]
        while stack:
            n = stack.pop()
            total += len(n.key)
            stack.extend(n.children.values())
        return total

    def _collect_evictable_leaves(self):
        leaves = []

        def dfs(node):
            if not node.children:
                if node.ref_count == 0 and node is not self.tree.root:
                    leaves.append(node)
                return
            for child in node.children.values():
                dfs(child)

        dfs(self.tree.root)
        return leaves

    def evict_if_needed(self):
        while self.total_tokens() > self.max_tokens:
            leaves = self._collect_evictable_leaves()
            if not leaves:
                break
            leaves.sort(key=lambda n: n.last_accessed)
            victim = leaves[0]

            parent = victim.parent
            if parent is not None:
                first_tok = victim.key[0]
                if first_tok in parent.children:
                    del parent.children[first_tok]

            curr = parent
            while (
                curr is not None
                and curr is not self.tree.root
                and len(curr.children) == 1
                and curr.ref_count == 0
            ):
                only_child = next(iter(curr.children.values()))
                curr.key.extend(only_child.key)
                curr.children = only_child.children
                curr.value = only_child.value
                for c in curr.children.values():
                    c.parent = curr
                break

    def insert_and_cache(self, tokens, request_id=None):
        matched_len, match_node, rem_tokens = self.tree.match_prefix(tokens)
        if match_node is not self.tree.root:
            self._update_access(match_node)

        inserted_node = self.tree.insert(tokens, value=request_id)
        self._update_access(inserted_node)
        self.evict_if_needed()
        return matched_len, inserted_node
