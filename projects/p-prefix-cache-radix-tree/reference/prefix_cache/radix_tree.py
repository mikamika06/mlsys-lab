from typing import Optional


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
        if not chain_hashes:
            return

        curr = self.root
        idx = 0
        n = len(chain_hashes)

        while idx < n:
            first_hash = chain_hashes[idx]
            if first_hash not in curr.children:
                new_node = RadixNode(tuple(chain_hashes[idx:]))
                new_node.block_ids = list(block_ids[idx:])
                new_node.last_accessed = access_time
                curr.children[first_hash] = new_node
                return

            child = curr.children[first_hash]
            child.last_accessed = access_time

            common_len = 0
            max_len = min(len(child.prefix), n - idx)
            while common_len < max_len and child.prefix[common_len] == chain_hashes[idx + common_len]:
                common_len += 1

            if common_len < len(child.prefix):
                split_node = RadixNode(child.prefix[:common_len])
                split_node.block_ids = child.block_ids[:common_len]
                split_node.last_accessed = access_time

                child.prefix = child.prefix[common_len:]
                child.block_ids = child.block_ids[common_len:]

                split_node.children[child.prefix[0]] = child
                curr.children[first_hash] = split_node
                child = split_node

            idx += common_len
            curr = child

        curr.last_accessed = access_time

    def match_prefix(self, chain_hashes: list[int], access_time: float = 0.0) -> tuple[list[int], int]:
        matched_blocks = []
        matched_count = 0
        curr = self.root
        idx = 0
        n = len(chain_hashes)

        while idx < n:
            first_hash = chain_hashes[idx]
            if first_hash not in curr.children:
                break

            child = curr.children[first_hash]
            common_len = 0
            max_len = min(len(child.prefix), n - idx)

            while common_len < max_len and child.prefix[common_len] == chain_hashes[idx + common_len]:
                common_len += 1

            if common_len < len(child.prefix):
                if common_len > 0:
                    matched_blocks.extend(child.block_ids[:common_len])
                    matched_count += common_len
                break

            matched_blocks.extend(child.block_ids)
            matched_count += len(child.prefix)
            child.last_accessed = access_time
            idx += common_len
            curr = child

        if matched_count > 0:
            self.root.last_accessed = access_time
        return matched_blocks, matched_count

    def evict_lru(self, num_blocks: int) -> int:
        evicted_blocks = 0

        while evicted_blocks < num_blocks:
            candidates = []

            def find_leaves(node: RadixNode, parent: Optional[RadixNode], key_in_parent: Optional[int]):
                if node.ref_count > 0:
                    return
                if not node.children:
                    if parent is not None:
                        candidates.append((node.last_accessed, node, parent, key_in_parent))
                    return

                for k, child in list(node.children.items()):
                    find_leaves(child, node, k)

            find_leaves(self.root, None, None)

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            _, target, parent, key_in_parent = candidates[0]

            blocks_to_remove = len(target.block_ids)
            evicted_blocks += blocks_to_remove

            del parent.children[key_in_parent]

            if len(parent.children) == 1 and parent is not self.root and parent.ref_count == 0:
                single_child = next(iter(parent.children.values()))
                parent.prefix = parent.prefix + single_child.prefix
                parent.block_ids = parent.block_ids + single_child.block_ids
                parent.children = single_child.children
                parent.last_accessed = max(parent.last_accessed, single_child.last_accessed)

        return evicted_blocks
