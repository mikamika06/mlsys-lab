import hashlib
import bisect
import numpy as np


class OracleConsistentHashRing:
    def __init__(self, replicas=None, num_tokens=100):
        self.num_tokens = num_tokens
        self.ring = []
        self.node_map = {}
        if replicas:
            for r in sorted(replicas):
                self.add_replica(r)

    def _hash(self, val):
        return int(hashlib.md5(str(val).encode("utf-8")).hexdigest(), 16)

    def add_replica(self, replica_id):
        for i in range(self.num_tokens):
            token_key = f"{replica_id}-tok-{i}"
            h = self._hash(token_key)
            bisect.insort(self.ring, h)
            self.node_map[h] = replica_id

    def get_replica(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]


def generate_routing_dataset(num_keys=2000, num_logs=5000):
    np.random.seed(42)
    keys = [f"key_{i}" for i in range(num_keys)]
    ring = OracleConsistentHashRing(["r1", "r2", "r3", "r4", "r5"], num_tokens=100)

    logs = []
    for _ in range(num_logs):
        if np.random.rand() < 0.35:
            r = "r3"
        else:
            k = np.random.choice(keys)
            r = ring.get_replica(k)
        logs.append({"key": np.random.choice(keys), "replica": r})

    return keys, logs
