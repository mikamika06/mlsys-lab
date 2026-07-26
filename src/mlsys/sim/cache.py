"""Deterministic set-associative cache simulator.
Runs identically on every machine (pure Python, no hardware). Tasks feed a trace
of byte addresses; the model returns exact hit/miss/eviction counts under LRU.
This is the MLSYS VIRTUAL CACHE — a pinned spec, the same for everyone.
"""
from collections import OrderedDict


class Cache:
    def __init__(self, line_bytes=64, sets=64, ways=8, policy="lru"):
        assert policy in ("lru", "fifo")
        self.line_bytes = line_bytes
        self.sets = sets
        self.ways = ways
        self.policy = policy
        self.reset()

    def reset(self):
        self._sets = [OrderedDict() for _ in range(self.sets)]  # set -> {tag: True} MRU last
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _index_tag(self, addr):
        block = addr // self.line_bytes
        return block % self.sets, block // self.sets

    def access(self, addr):
        """One access to a byte address. Returns True on hit, False on miss."""
        idx, tag = self._index_tag(addr)
        s = self._sets[idx]
        if tag in s:
            self.hits += 1
            if self.policy == "lru":
                s.move_to_end(tag)
            return True
        self.misses += 1
        if len(s) >= self.ways:
            s.popitem(last=False)  # evict LRU/FIFO (oldest)
            self.evictions += 1
        s[tag] = True
        return False

    def run(self, addresses):
        """Feed an iterable of byte addresses; return the summary dict."""
        for a in addresses:
            self.access(int(a))
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "accesses": total, "miss_rate": (self.misses / total) if total else 0.0}


def simulate(addresses, line_bytes=64, sets=64, ways=8, policy="lru"):
    """Convenience: build a fresh cache and run the trace. Deterministic."""
    return Cache(line_bytes, sets, ways, policy).run(addresses)
