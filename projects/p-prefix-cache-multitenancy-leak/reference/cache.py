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
        if not self.isolation:
            return hash((parent_id, tuple(tokens)))
        return hash((tenant_id, parent_id, tuple(tokens)))

    def insert(self, tokens, tenant_id, is_system=False):
        parent_id = 0
        allocated = []
        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i:i + self.block_size]
            if len(chunk) < self.block_size:
                break

            sys_key = self.compute_key("system", parent_id, chunk) if self.shared_system else None
            usr_key = self.compute_key(tenant_id, parent_id, chunk)

            if sys_key is not None and sys_key in self.hash_to_block:
                parent_id = self.hash_to_block[sys_key]
            elif usr_key in self.hash_to_block:
                parent_id = self.hash_to_block[usr_key]
            else:
                new_block = self.allocator.alloc()
                if new_block is None:
                    break
                eff_tenant = "system" if (self.shared_system and is_system) else tenant_id
                key = self.compute_key(eff_tenant, parent_id, chunk)
                self.hash_to_block[key] = new_block
                self.block_to_hash[new_block] = key
                parent_id = new_block
            allocated.append(parent_id)
        return allocated

    def match(self, tokens, tenant_id):
        parent_id = 0
        matched = []
        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i:i + self.block_size]
            if len(chunk) < self.block_size:
                break

            sys_key = self.compute_key("system", parent_id, chunk) if self.shared_system else None
            usr_key = self.compute_key(tenant_id, parent_id, chunk)

            if sys_key is not None and sys_key in self.hash_to_block:
                parent_id = self.hash_to_block[sys_key]
                matched.append(parent_id)
            elif usr_key in self.hash_to_block:
                parent_id = self.hash_to_block[usr_key]
                matched.append(parent_id)
            else:
                break
        return matched
