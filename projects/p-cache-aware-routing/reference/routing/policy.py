import numpy as np
from routing.affinity import compute_affinity


class CacheAwarePolicy:
    def __init__(self, num_replicas, load_weight=0.3):
        self.num_replicas = num_replicas
        self.load_weight = load_weight
        self.loads = np.zeros(num_replicas, dtype=int)

    def route(self, prompt, replica_states):
        scores = []
        for i in range(self.num_replicas):
            aff = compute_affinity(prompt, replica_states[i])
            load_penalty = self.load_weight * (self.loads[i] / (1.0 + np.max(self.loads)))
            score = aff - load_penalty
            scores.append(score)
        best_replica = int(np.argmax(scores))
        self.loads[best_replica] += 1
        return best_replica

    def release(self, replica_id):
        if self.loads[replica_id] > 0:
            self.loads[replica_id] -= 1
