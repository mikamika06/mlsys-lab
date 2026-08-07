class BlockAllocator:
    def __init__(self, capacity=10000):
        self.free_list = list(range(1, capacity + 1))
        self.used = 0

    def alloc(self):
        if not self.free_list:
            return None
        self.used += 1
        return self.free_list.pop(0)

def get_trace():
    sys = list(range(100, 140))
    return [
        ("A", sys, True),
        ("A", sys + list(range(200, 220)), False),
        ("B", sys, True),
        ("B", sys + list(range(300, 320)), False),
        ("C", sys + list(range(400, 420)), False),
    ]

def run_trace(cache, trace):
    hits = 0
    total_blocks = 0
    for tenant, tokens, is_system in trace:
        matched = cache.match(tokens, tenant)
        hits += len(matched)
        total_blocks += len(tokens) // cache.block_size
        cache.insert(tokens, tenant, is_system)
    return hits, total_blocks
