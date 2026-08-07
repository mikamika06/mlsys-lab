class RadixTreeNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.children = {}
        self.ref_count = 1
        self.parent = None


class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode(b"")

    def insert(self, token_ids, block_ids):
        curr = self.root
        curr.ref_count += 1
        idx = 0
        while idx < len(token_ids):
            matched_child = None
            for k, child in curr.children.items():
                common = 0
                for a, b in zip(k, token_ids[idx:]):
                    if a == b:
                        common += 1
                    else:
                        break
                if common > 0:
                    matched_child = (k, child, common)
                    break
            if matched_child is None:
                new_node = RadixTreeNode(token_ids[idx:], block_ids[idx:])
                new_node.parent = curr
                new_node.ref_count = 1
                curr.children[token_ids[idx:]] = new_node
                return

            k, child, common = matched_child
            if common < len(k):
                prefix_key = k[:common]
                suffix_key = k[common:]

                mid_node = RadixTreeNode(prefix_key, child.value[:common] if child.value else None)
                mid_node.parent = curr
                mid_node.ref_count = child.ref_count
                mid_node.children[suffix_key] = child

                child.parent = mid_node
                child.key = suffix_key
                if child.value:
                    child.value = child.value[common:]

                curr.children.pop(k)
                curr.children[prefix_key] = mid_node

                new_node = RadixTreeNode(token_ids[idx + common:], block_ids[idx + common:])
                new_node.parent = mid_node
                new_node.ref_count = 1
                mid_node.children[token_ids[idx + common:]] = new_node
                mid_node.ref_count += 1
                return
            else:
                curr = child
                curr.ref_count += 1
                idx += common

    def match(self, token_ids):
        curr = self.root
        matched_blocks = []
        idx = 0
        while idx < len(token_ids):
            found = False
            for k, child in curr.children.items():
                if token_ids[idx:idx + len(k)] == k:
                    matched_blocks.extend(child.value if child.value else [])
                    curr = child
                    idx += len(k)
                    found = True
                    break
            if not found:
                break
        return matched_blocks, idx

    def decref(self, node):
        curr = node
        while curr is not None and curr != self.root:
            curr.ref_count -= 1
            if curr.ref_count <= 0:
                if curr.parent and curr.key in curr.parent.children:
                    curr.parent.children.pop(curr.key)
            curr = curr.parent
