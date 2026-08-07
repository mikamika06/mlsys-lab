class CacheSimulator:
    def __init__(self, capacity, policy="lru"):
        raise NotImplementedError

    def access(self, key, compute_cost=1.0):
        raise NotImplementedError

    def get_stats(self):
        raise NotImplementedError
