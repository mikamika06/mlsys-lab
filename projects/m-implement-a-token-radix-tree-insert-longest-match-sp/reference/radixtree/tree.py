class RadixNode:
    def __init__(self, prefix=None):
        self.prefix = prefix if prefix is not None else []
        self.children = {}
        self.value = None

class TokenRadixTree:
    def __init__(self):
        self.root = RadixNode([])

    def insert(self, tokens, value=None):
        curr = self.root
        idx = 0
        while idx < len(tokens):
            found = None
            for first_tok, child in curr.children.items():
                if child.prefix and child.prefix[0] == tokens[idx]:
                    found = child
                    break
            if found is None:
                new_node = RadixNode(tokens[idx:])
                new_node.value = value
                curr.children[tokens[idx]] = new_node
                return

            common = 0
            while common < len(found.prefix) and (idx + common) < len(tokens) and found.prefix[common] == tokens[idx + common]:
                common += 1

            if common == len(found.prefix):
                idx += common
                curr = found
                if idx == len(tokens):
                    curr.value = value
            else:
                split_node = RadixNode(found.prefix[common:])
                split_node.children = found.children
                split_node.value = found.value

                found.prefix = found.prefix[:common]
                found.children = {split_node.prefix[0]: split_node}
                found.value = None

                if idx + common == len(tokens):
                    found.value = value
                    curr = found
                else:
                    new_node = RadixNode(tokens[idx + common:])
                    new_node.value = value
                    found.children[new_node.prefix[0]] = new_node
                    curr = found
                break

    def longest_match(self, tokens):
        curr = self.root
        idx = 0
        matched_tokens = []
        while idx < len(tokens):
            found = None
            for first_tok, child in curr.children.items():
                if child.prefix and child.prefix[0] == tokens[idx]:
                    found = child
                    break
            if found is None:
                break
            common = 0
            while common < len(found.prefix) and (idx + common) < len(tokens) and found.prefix[common] == tokens[idx + common]:
                common += 1
            if common < len(found.prefix):
                matched_tokens.extend(found.prefix[:common])
                break
            matched_tokens.extend(found.prefix)
            idx += common
            curr = found
        return matched_tokens, curr.value if idx == len(tokens) or curr != self.root else None
