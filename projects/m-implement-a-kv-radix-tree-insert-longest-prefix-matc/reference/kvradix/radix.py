class RadixNode:
    """Radix tree node holding token sequences and KV references."""

    def __init__(self, key=None):
        self.key = list(key) if key is not None else []
        self.children = {}
        self.parent = None
        self.value = None
        self.ref_count = 0
        self.last_accessed = 0.0


class RadixTree:
    """Radix Tree structure supporting insert and longest-prefix match."""

    def __init__(self):
        self.root = RadixNode()

    def match_prefix(self, tokens):
        curr = self.root
        tokens = list(tokens)
        idx = 0
        matched_len = 0

        while idx < len(tokens):
            first_tok = tokens[idx]
            if first_tok not in curr.children:
                break
            child = curr.children[first_tok]
            key = child.key
            k_len = len(key)

            match_k = 0
            while match_k < k_len and (idx + match_k) < len(tokens):
                if key[match_k] == tokens[idx + match_k]:
                    match_k += 1
                else:
                    break

            matched_len += match_k
            idx += match_k

            if match_k == k_len:
                curr = child
            else:
                break

        return matched_len, curr, tokens[idx:]

    def insert(self, tokens, value=None):
        tokens = list(tokens)
        if not tokens:
            return self.root

        curr = self.root
        idx = 0

        while idx < len(tokens):
            first_tok = tokens[idx]
            if first_tok not in curr.children:
                new_node = RadixNode(tokens[idx:])
                new_node.parent = curr
                new_node.value = value
                curr.children[first_tok] = new_node
                return new_node

            child = curr.children[first_tok]
            key = child.key
            k_len = len(key)

            match_k = 0
            while match_k < k_len and (idx + match_k) < len(tokens):
                if key[match_k] == tokens[idx + match_k]:
                    match_k += 1
                else:
                    break

            if match_k < k_len:
                split_node = RadixNode(key[:match_k])
                split_node.parent = curr
                curr.children[first_tok] = split_node

                child.key = key[match_k:]
                child.parent = split_node
                split_node.children[child.key[0]] = child

                if idx + match_k == len(tokens):
                    split_node.value = value
                    return split_node

                rem_tokens = tokens[idx + match_k:]
                new_node = RadixNode(rem_tokens)
                new_node.parent = split_node
                new_node.value = value
                split_node.children[rem_tokens[0]] = new_node
                return new_node
            else:
                idx += k_len
                curr = child

        curr.value = value
        return curr
