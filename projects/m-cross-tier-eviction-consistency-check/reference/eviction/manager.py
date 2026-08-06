"""Cross-tier eviction manager implementation."""

from eviction.checker import check_cross_tier_consistency

class CrossTierEvictionManager:
    def __init__(self, t0_capacity, t1_capacity):
        self.t0_capacity = t0_capacity
        self.t1_capacity = t1_capacity
        self.t0_state = {}
        self.t1_state = {}

    def register_block(self, block_id, data_hash, tier=0):
        if tier == 0:
            if len(self.t0_state) >= self.t0_capacity and block_id not in self.t0_state:
                raise RuntimeError("T0 capacity exceeded")
            self.t0_state[block_id] = {"hash": data_hash, "dirty": False}
        elif tier == 1:
            if len(self.t1_state) >= self.t1_capacity and block_id not in self.t1_state:
                raise RuntimeError("T1 capacity exceeded")
            self.t1_state[block_id] = {"hash": data_hash, "stale": False, "status": "VALID"}
        else:
            raise ValueError("Invalid tier")

    def evict_from_t0(self, block_id, preserve_in_t1=True):
        if block_id not in self.t0_state:
            return False
        meta = self.t0_state.pop(block_id)
        if preserve_in_t1:
            if len(self.t1_state) >= self.t1_capacity and block_id not in self.t1_state:
                lru_key = next(iter(self.t1_state))
                del self.t1_state[lru_key]
            self.t1_state[block_id] = {
                "hash": meta["hash"],
                "stale": False,
                "status": "VALID"
            }
        else:
            if block_id in self.t1_state:
                del self.t1_state[block_id]
        return True

    def get_tier_states(self):
        return self.t0_state, self.t1_state
