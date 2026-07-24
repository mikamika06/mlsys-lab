class _TrieNode:
    __slots__ = ("children", "is_word")
    def __init__(self):
        self.children = {}
        self.is_word = False

def process_ops(ops):
    root = _TrieNode()
    reuse_lengths = []
    recompute_counts = []

    for op, word in ops:
        node = root
        i = 0
        # walk as far as we can match existing prefixes
        while i < len(word) and word[i] in node.children:
            node = node.children[word[i]]
            i += 1

        reuse_lengths.append(i)
        recompute_counts.append(len(word) - i)

        if op == 'add':
            # insert the remaining suffix into the trie
            while i < len(word):
                new_node = _TrieNode()
                node.children[word[i]] = new_node
                node = new_node
                i += 1
            node.is_word = True

    return reuse_lengths, recompute_counts
