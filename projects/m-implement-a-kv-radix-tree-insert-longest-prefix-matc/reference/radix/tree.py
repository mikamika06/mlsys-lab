import time


class RadixTreeNode:
    def __init__(self, key):
        self.key = tuple(key)
        self.children = {}
        self.value = None
        self.ref_count = 0
        self.last_access = time.time()


class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode(())

    def insert(self, tokens, value):
        tokens = tuple(tokens)
        curr = self.root
        idx = 0
        while idx < len(tokens):
            matched_child = None
            for k, child in curr.children.items():
                match_len = 0
                for a, b in zip(k, tokens[idx:]):
                    if a == b:
                        match_len += 1
                    else:
                        break
                if match_len > 0:
                    matched_child = (k, child, match_len)
                    break

            if not matched_child:
                new_node = RadixTreeNode(tokens[idx:])
                new_node.value = value
                new_node.ref_count = 1
                curr.children[tokens[idx:]] = new_node
                return new_node

            k, child, match_len = matched_child
            if match_len == len(k):
                curr = child
                idx += match_len
            else:
                prefix = k[:match_len]
                suffix = k[match_len:]

                new_child = RadixTreeNode(suffix)
                new_child.children = child.children
                new_child.value = child.value
                new_child.ref_count = child.ref_count
                new_child.last_access = child.last_access

                child.key = prefix
                child.children = {suffix: new_child}
                child.value = None

                curr.children.pop(k)
                curr.children[prefix] = child

                curr = child
                idx += match_len

        curr.value = value
        curr.ref_count += 1
        return curr

    def match_prefix(self, tokens):
        tokens = tuple(tokens)
        curr = self.root
        matched_tokens = []
        idx = 0
        while idx < len(tokens):
            found = False
            for k, child in curr.children.items():
                match_len = 0
                for a, b in zip(k, tokens[idx:]):
                    if a == b:
                        match_len += 1
                    else:
                        break
                if match_len > 0:
                    if match_len == len(k):
                        curr = child
                        matched_tokens.extend(k)
                        idx += match_len
                        found = True
                        break
                    else:
                        matched_tokens.extend(k[:match_len])
                        idx += match_len
                        curr = child
                        found = True
                        break
            if not found:
                break
        return matched_tokens, curr
