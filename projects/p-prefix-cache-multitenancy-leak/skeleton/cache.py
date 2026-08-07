import ref

class PrefixCache:
    def __init__(self, block_size, allocator, isolation=False, shared_system=False):
        self.block_size = block_size
        self.allocator = allocator
        self.isolation = isolation
        self.shared_system = shared_system
        self.hash_to_block = {}
        self.block_to_hash = {}

    def compute_key(self, tenant_id, parent_id, tokens):
        raise NotImplementedError

    def insert(self, tokens, tenant_id, is_system=False):
        raise NotImplementedError

    def match(self, tokens, tenant_id):
        raise NotImplementedError
