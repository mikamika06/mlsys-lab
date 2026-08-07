import numpy as np

class MoEOffloader:
    def __init__(self, tensor_sizes, memory_limit):
        self.tensor_sizes = tensor_sizes
        self.memory_limit = memory_limit

    def measure_frequencies(self, traces):
        n_tensors = len(self.tensor_sizes)
        counts = np.zeros(n_tensors, dtype=np.float64)
        if not traces:
            return counts
        for trace in traces:
            for idx in trace:
                if 0 <= idx < n_tensors:
                    counts[idx] += 1.0
        return counts / len(traces)

    def compute_rules(self, frequencies, memory_budget):
        sizes = np.array(self.tensor_sizes, dtype=np.float64)
        total_mem = np.sum(sizes)
        indexed = sorted(enumerate(frequencies), key=lambda x: x[1])
        offloaded = set()
        current_mem = total_mem
        for idx, _ in indexed:
            if current_mem <= memory_budget:
                break
            offloaded.add(idx)
            current_mem -= sizes[idx]
        return offloaded

    def evaluate_latency(self, offloaded, base_latency, penalty_factor=2.0):
        lat = np.array(base_latency, dtype=np.float64)
        for idx in offloaded:
            lat[idx] *= penalty_factor
        return float(np.sum(lat))

    def verify_output(self, ref_out, cand_out, tol=1e-5):
        return bool(np.allclose(ref_out, cand_out, atol=tol))

    def check_constraints(self, offloaded, memory_budget, latency, max_latency):
        sizes = np.array(self.tensor_sizes, dtype=np.float64)
        current_mem = np.sum(sizes) - np.sum(sizes[list(offloaded)]) if offloaded else np.sum(sizes)
        return bool(current_mem <= memory_budget and latency <= max_latency)
