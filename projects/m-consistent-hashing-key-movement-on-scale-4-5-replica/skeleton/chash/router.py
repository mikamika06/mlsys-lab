class ConsistentHashRing:
    def __init__(self, replicas=None, num_tokens=100):
        raise NotImplementedError

    def add_replica(self, replica_id):
        raise NotImplementedError

    def remove_replica(self, replica_id):
        raise NotImplementedError

    def get_replica(self, key):
        raise NotImplementedError


def calculate_remapping_fraction(ring_before, ring_after, sample_keys):
    raise NotImplementedError
