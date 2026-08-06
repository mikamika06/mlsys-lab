from radix.tree import TokenRadixTree

class FlatCache:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.cache = set()

    def access(self, tokens):
        key = tuple(tokens)
        if key in self.cache:
            return True
        if len(self.cache) >= self.capacity:
            self.cache.pop()
        self.cache.add(key)
        return False

class RadixCache:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.tree = TokenRadixTree()
        self.count = 0

    def access(self, tokens):
        match_tokens, val = self.tree.longest_match(tokens)
        hit = len(match_tokens) == len(tokens)
        if not hit:
            if self.count >= self.capacity:
                self.tree = TokenRadixTree()
                self.count = 0
            self.tree.insert(tokens, value=True)
            self.count += 1
        return hit
