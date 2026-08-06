import hashlib
import bisect


class ConsistentHashRing:
    def __init__(self, replicas=None, num_tokens=100):
        """Consistent hash ring using virtual nodes."""
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

    def remove_replica(self, replica_id):
        to_remove = [h for h, r in self.node_map.items() if r == replica_id]
        for h in to_remove:
            idx = bisect.bisect_left(self.ring, h)
            if idx < len(self.ring) and self.ring[idx] == h:
                self.ring.pop(idx)
            del self.node_map[h]

    def get_replica(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.node_map[self.ring[idx]]


def calculate_remapping_fraction(ring_before, ring_after, sample_keys):
    moved = 0
    for key in sample_keys:
        if ring_before.get_replica(key) != ring_after.get_replica(key):
            moved += 1
    return moved / len(sample_keys) if sample_keys else 0.0
