class RadixEvictor:
    def __init__(self, tree, max_blocks):
        self.tree = tree
        self.max_blocks = max_blocks
        self.lru_order = []
        self.evicted_blocks = []

    def on_insert(self, block_ids):
        for b in block_ids:
            self.lru_order.append(b)
        while len(self.lru_order) > self.max_blocks:
            self.evict()

    def evict(self):
        if not self.lru_order:
            return None
        evicted = self.lru_order.pop(0)
        self.evicted_blocks.append(evicted)
        return evicted
