import numpy as np

class RingBalancer:
    def __init__(self, world_size, seq_len):
        self.world_size = world_size
        self.seq_len = seq_len

    def get_workload_per_step(self):
        steps = self.world_size
        workloads = []
        block_size = self.seq_len // self.world_size
        for step in range(steps):
            step_load = []
            for rank in range(self.world_size):
                kv_rank = (rank - step) % self.world_size
                if kv_rank <= rank:
                    load = block_size * block_size
                else:
                    load = 0
                step_load.append(load)
            workloads.append(step_load)
        return workloads

    def rebalance(self, workloads):
        flat = []
        for s_idx, step in enumerate(workloads):
            for r_idx, load in enumerate(step):
                flat.append((load, s_idx, r_idx))
        flat.sort(key=lambda x: x[0], reverse=True)
        new_alloc = [[0]*self.world_size for _ in range(self.world_size)]
        rank_loads = [0]*self.world_size
        for item in flat:
            load, s_idx, r_idx = item
            min_rank = int(np.argmin(rank_loads))
            new_alloc[s_idx][min_rank] += load
            rank_loads[min_rank] += load
        return new_alloc

    def verify_equivalence(self, orig_output, new_output):
        return np.allclose(orig_output, new_output, atol=1e-5)

    def measure_utilization(self, workloads):
        total_slots = len(workloads) * self.world_size * max(max(w) for w in workloads)
        if total_slots == 0:
            return 0.0
        active = sum(sum(w) for w in workloads)
        return active / total_slots

    def check_imbalance_threshold(self, workloads, threshold=0.6):
        max_rank_load = max(sum(step[r] for step in workloads) for r in range(self.world_size))
        total = sum(sum(step) for step in workloads)
        avg = total / self.world_size if self.world_size > 0 else 0
        imbalance = (max_rank_load - avg) / (avg + 1e-8)
        return imbalance < threshold

    def predict_scaling(self, num_devices):
        base_time = 100.0
        return base_time / num_devices + 5.0 * np.log(num_devices)
