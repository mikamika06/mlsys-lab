"""Cache simulation with LRU, LFU, and Cost-Aware eviction strategies."""


class CacheSimulator:
    def __init__(self, capacity, policy="lru"):
        self.capacity = capacity
        self.policy = policy
        self.cache = {}
        self.freqs = {}
        self.costs = {}
        self.access_time = {}
        self.timer = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def access(self, key, compute_cost=1.0):
        self.timer += 1
        if key in self.cache:
            self.hits += 1
            self.freqs[key] += 1
            self.access_time[key] = self.timer
            return True

        self.misses += 1
        if len(self.cache) >= self.capacity and self.capacity > 0:
            self._evict()

        if self.capacity > 0:
            self.cache[key] = True
            self.freqs[key] = 1
            self.costs[key] = compute_cost
            self.access_time[key] = self.timer
        return False

    def _evict(self):
        if not self.cache:
            return

        victim = None
        if self.policy == "lru":
            victim = min(self.cache.keys(), key=lambda k: self.access_time[k])
        elif self.policy == "lfu":
            victim = min(self.cache.keys(), key=lambda k: (self.freqs[k], self.access_time[k]))
        elif self.policy == "cost":
            victim = min(self.cache.keys(), key=lambda k: (self.costs[k], -self.access_time[k]))
        else:
            victim = next(iter(self.cache.keys()))

        del self.cache[victim]
        del self.freqs[victim]
        del self.costs[victim]
        del self.access_time[victim]
        self.evictions += 1

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }
