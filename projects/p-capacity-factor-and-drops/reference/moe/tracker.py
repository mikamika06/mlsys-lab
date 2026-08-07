class DropTracker:
    def __init__(self, capacity_factor: float):
        self.cf = capacity_factor
        self.total_tokens = 0
        self.dropped_tokens = 0

    def update(self, routings, expert_capacities):
        import numpy as np
        routings = np.asarray(routings)
        for e_idx, cap in enumerate(expert_capacities):
            assigned = np.sum(routings == e_idx)
            self.total_tokens += assigned
            if assigned > cap:
                self.dropped_tokens += (assigned - cap)

    def drop_ratio(self):
        if self.total_tokens == 0:
            return 0.0
        return float(self.dropped_tokens) / float(self.total_tokens)
