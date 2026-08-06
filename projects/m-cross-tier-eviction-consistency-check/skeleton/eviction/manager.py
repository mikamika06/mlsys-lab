"""Cross-tier eviction manager."""

class CrossTierEvictionManager:
    def __init__(self, t0_capacity, t1_capacity):
        raise NotImplementedError

    def register_block(self, block_id, data_hash, tier=0):
        raise NotImplementedError

    def evict_from_t0(self, block_id, preserve_in_t1=True):
        raise NotImplementedError

    def get_tier_states(self):
        raise NotImplementedError
