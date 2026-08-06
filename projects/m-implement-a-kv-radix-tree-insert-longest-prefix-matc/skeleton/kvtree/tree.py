class RadixNode:
    def __init__(self, key_tokens=None, block_ids=None):
        raise NotImplementedError

class RadixTree:
    def __init__(self):
        raise NotImplementedError

    def insert(self, tokens, block_ids):
        raise NotImplementedError

    def match_prefix(self, tokens):
        raise NotImplementedError
