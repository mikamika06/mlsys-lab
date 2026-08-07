class PrefixCache:
    def __init__(self, capacity=100, isolate=True):
        self.capacity = capacity
        self.isolate = isolate
        self.cache = set()

    def insert(self, tokens, tenant_id="default", system_prefixes=None):
        if system_prefixes is None:
            system_prefixes = []
        curr = []
        for t in tokens:
            curr.append(t)
            is_sys = any(tuple(curr[-len(sp):]) == tuple(sp) for sp in system_prefixes)
            if self.isolate and not is_sys:
                key = (tenant_id, tuple(curr))
            else:
                key = tuple(curr)
            h = hash(key)
            if len(self.cache) >= self.capacity:
                self.cache.pop()
            self.cache.add(h)

    def lookup(self, tokens, tenant_id="default", system_prefixes=None):
        if system_prefixes is None:
            system_prefixes = []
        curr = []
        hits = 0
        for t in tokens:
            curr.append(t)
            is_sys = any(tuple(curr[-len(sp):]) == tuple(sp) for sp in system_prefixes)
            if self.isolate and not is_sys:
                key = (tenant_id, tuple(curr))
            else:
                key = tuple(curr)
            h = hash(key)
            if h in self.cache:
                hits += 1
        return hits
