class RadixNode:
    def __init__(self, prefix=None):
        self.prefix = prefix if prefix is not None else []
        self.children = {}
        self.value = None

class TokenRadixTree:
    def __init__(self):
        self.root = RadixNode(prefix=[])

    def insert(self, tokens, value=None):
        curr = self.root
        idx = 0
        while idx < len(tokens):
            matched_child = None
            for token, child in curr.children.items():
                if token == tokens[idx]:
                    matched_child = child
                    break
            if not matched_child:
                new_node = RadixNode(prefix=tokens[idx:])
                new_node.value = value
                curr.children[tokens[idx]] = new_node
                return

            prefix = matched_child.prefix
            common_len = 0
            while common_len < len(prefix) and (idx + common_len) < len(tokens) and prefix[common_len] == tokens[idx + common_len]:
                common_len += 1

            if common_len == len(prefix):
                idx += common_len
                curr = matched_child
            else:
                split_node = RadixNode(prefix=prefix[:common_len])
                remainder_child = RadixNode(prefix=prefix[common_len:])
                remainder_child.children = matched_child.children
                remainder_child.value = matched_child.value

                if idx + common_len < len(tokens):
                    new_branch = RadixNode(prefix=tokens[idx + common_len:])
                    new_branch.value = value
                    split_node.children = {
                        prefix[common_len]: remainder_child,
                        tokens[idx + common_len]: new_branch
                    }
                else:
                    split_node.value = value
                    split_node.children = {prefix[common_len]: remainder_child}

                curr.children[tokens[idx]] = split_node
                return
        curr.value = value

    def longest_match(self, tokens):
        curr = self.root
        idx = 0
        matched_tokens = []
        while idx < len(tokens):
            token = tokens[idx]
            if token not in curr.children:
                break
            child = curr.children[token]
            prefix = child.prefix
            common_len = 0
            while common_len < len(prefix) and (idx + common_len) < len(tokens) and prefix[common_len] == tokens[idx + common_len]:
                common_len += 1
            if common_len < len(prefix):
                break
            matched_tokens.extend(prefix)
            idx += common_len
            curr = child
        return matched_tokens, curr.value
