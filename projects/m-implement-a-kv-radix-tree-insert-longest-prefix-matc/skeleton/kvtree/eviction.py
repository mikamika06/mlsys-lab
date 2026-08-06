class RadixEvictor:
    def __init__(self, tree, max_blocks):
        raise NotImplementedError

    def on_insert(self, block_ids):
        raise NotImplementedError

    def evict(self):
        raise NotImplementedError
