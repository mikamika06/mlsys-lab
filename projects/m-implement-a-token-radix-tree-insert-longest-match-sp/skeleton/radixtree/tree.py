class RadixNode:
    def __init__(self, prefix=None):
        raise NotImplementedError

class TokenRadixTree:
    def __init__(self):
        raise NotImplementedError

    def insert(self, tokens, value=None):
        raise NotImplementedError

    def longest_match(self, tokens):
        raise NotImplementedError
