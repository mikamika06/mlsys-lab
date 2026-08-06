class RadixNode:
    def __init__(self, key_tokens=None, block_ids=None):
        self.key_tokens = list(key_tokens) if key_tokens else []
        self.block_ids = list(block_ids) if block_ids else []
        self.children = {}
        self.parent = None
        self.ref_count = 1
        self.is_leaf = True

class RadixTree:
    def __init__(self):
        self.root = RadixNode([], [])
        self.leaves = set()

    def insert(self, tokens, block_ids):
        curr = self.root
        idx = 0
        while idx < len(tokens):
            matched_child = None
            for ch_token, child in curr.children.items():
                if ch_token == tokens[idx]:
                    matched_child = child
                    break
            if not matched_child:
                new_node = RadixNode(tokens[idx:], block_ids[idx:])
                new_node.parent = curr
                curr.children[tokens[idx]] = new_node
                if curr in self.leaves:
                    self.leaves.remove(curr)
                    curr.is_leaf = False
                self.leaves.add(new_node)
                new_node.is_leaf = True
                return

            common = 0
            limit = min(len(matched_child.key_tokens), len(tokens) - idx)
            while common < limit and matched_child.key_tokens[common] == tokens[idx + common]:
                common += 1

            if common < len(matched_child.key_tokens):
                split_node = RadixNode(matched_child.key_tokens[common:], matched_child.block_ids[common:])
                split_node.parent = matched_child
                split_node.ref_count = matched_child.ref_count
                split_node.children = matched_child.children
                for _, ch in split_node.children.items():
                    ch.parent = split_node

                matched_child.key_tokens = matched_child.key_tokens[:common]
                matched_child.block_ids = matched_child.block_ids[:common]
                matched_child.children = {split_node.key_tokens[0]: split_node}
                matched_child.is_leaf = False
                if matched_child in self.leaves:
                    self.leaves.remove(matched_child)

                if common == len(tokens) - idx:
                    matched_child.block_ids = block_ids[idx:]
                    matched_child.ref_count += 1
                    if not matched_child.children:
                        self.leaves.add(matched_child)
                        matched_child.is_leaf = True
                    return
                else:
                    new_node = RadixNode(tokens[idx + common:], block_ids[idx + common:])
                    new_node.parent = matched_child
                    matched_child.children[new_node.key_tokens[0]] = new_node
                    self.leaves.add(new_node)
                    new_node.is_leaf = True
                    return
            else:
                idx += common
                if idx == len(tokens):
                    matched_child.ref_count += 1
                    return
                curr = matched_child

    def match_prefix(self, tokens):
        curr = self.root
        matched_blocks = []
        idx = 0
        while idx < len(tokens):
            matched_child = None
            for ch_token, child in curr.children.items():
                if ch_token == tokens[idx]:
                    matched_child = child
                    break
            if not matched_child:
                break
            common = 0
            limit = min(len(matched_child.key_tokens), len(tokens) - idx)
            while common < limit and matched_child.key_tokens[common] == tokens[idx + common]:
                common += 1
            if common < len(matched_child.key_tokens):
                break
            matched_blocks.extend(matched_child.block_ids)
            idx += common
            curr = matched_child
        return matched_blocks, idx
