import numpy as np

class ActivationTracker:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.counts = np.zeros(num_experts, dtype=np.int64)

    def update(self, selected_experts):
        for e in selected_experts:
            self.counts[e] += 1

    def get_distribution(self):
        total = self.counts.sum()
        if total == 0:
            return np.ones(self.num_experts, dtype=np.float64) / self.num_experts
        return self.counts.astype(np.float64) / total
