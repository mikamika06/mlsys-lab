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
        raise NotImplementedError

    def match(self, token_ids):
        raise NotImplementedError

    def decref(self, node):
        raise NotImplementedError
